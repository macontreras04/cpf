"""Part 2: direct multi-horizon quantile forecast of savings balances with XGBoost.

For each horizon h = 1..12 months (21 business days each), one XGBoost model predicts the
5%, 50% and 95% quantiles of log(B_{t+h} / B_t). Backtest: annual walk-forward 2019-2025
with purging, against a no-change forecast, a historical drift and historical quantile bands.

Usage:
    python forecast_part2.py --input data/savings_deposits_indexed.csv
"""
import argparse, json, numpy as np, pandas as pd, xgboost as xgb
import core_noncore_model as M          # reuses load_data, clean_data and build_features
ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); args=ap.parse_args()
SEED=42; Q=[0.05,0.5,0.95]; H=[21*k for k in range(1,13)]  # 1..12 months ahead (business days)
FEAT=["ret_5","ret_21","ret_63","ret_126","ret_252","vol_21","vol_63","vol_252","vol_ewma",
      "dev_ma_63","dev_ma_252","dd_252","up_252","worst_21d_252","slope_63","month","quarter_end","month_end"]
def model(h):
    return xgb.XGBRegressor(objective="reg:quantileerror",quantile_alpha=np.array(Q),max_depth=2,n_estimators=300,
        learning_rate=0.03,subsample=0.7,colsample_bytree=0.6,reg_lambda=5.0,min_child_weight=max(40,h),
        tree_method="hist",random_state=SEED,n_jobs=-1)
def pin(y,q,a): d=y-q; return np.mean(np.maximum(a*d,(a-1)*d))
d=M.clean_data(M.load_data(args.input))[0].set_index('date'); OUT={}
for key in ['sight','other']:
    s=d[key]; ls=np.log(s); X=M.build_features(s)[FEAT]; ok=X.notna().all(axis=1)
    pos=pd.Series(np.arange(len(s)),index=s.index); rows=[]
    for h in H:
        y=ls.shift(-h)-ls
        for yr in range(2019,2026):
            ft=pos[s.index.year==yr].min()
            tr=ok&y.notna()&(pos+h<ft); te=ok&y.notna()&(s.index.year==yr)
            if te.sum()==0: continue
            P=np.sort(model(h).fit(X[tr],y[tr]).predict(X[te]).reshape(-1,3),axis=1)
            drift=y[tr].mean(); hsq=y[tr].quantile([.05,.95]).values
            rows.append(pd.DataFrame(dict(h=h,yr=yr,y=y[te].values,q05=P[:,0],q50=P[:,1],q95=P[:,2],drift=drift,hs05=hsq[0],hs95=hsq[1]),index=s.index[te]))
    B=pd.concat(rows)
    ev=[]
    for h,g in B.groupby('h'):
        ev.append(dict(h=h,mae_xgb=np.mean(abs(g.y-g.q50)),mae_rw=np.mean(abs(g.y)),mae_drift=np.mean(abs(g.y-g.drift)),
          cover_xgb=np.mean((g.y>=g.q05)&(g.y<=g.q95)),cover_hs=np.mean((g.y>=g.hs05+g.drift*0)&(g.y<=g.hs95)),
          width_xgb=np.mean(g.q95-g.q05),width_hs=np.mean(g.hs95-g.hs05),
          below5=np.mean(g.y<g.q05),
          pin_xgb=np.mean([pin(g.y,g.q05,.05),pin(g.y,g.q95,.95)]),pin_hs=np.mean([pin(g.y,g.hs05,.05),pin(g.y,g.hs95,.95)])))
    E=pd.DataFrame(ev); E['skill_rw']=1-E.mae_xgb/E.mae_rw; E['skill_drift']=1-E.mae_xgb/E.mae_drift; E['skill_band']=1-E.pin_xgb/E.pin_hs
    print(key); print(E.round(3).to_string())
    byyr=B[B.h==252].groupby('yr').apply(lambda g: pd.Series(dict(mae_xgb=np.mean(abs(g.y-g.q50)),mae_rw=np.mean(abs(g.y)),mae_dr=np.mean(abs(g.y-g.drift)),cov=np.mean((g.y>=g.q05)&(g.y<=g.q95)))))
    print(byyr.round(3))
    # production forecast from last date
    last=s.index[-1]; xl=X.loc[[last]]; fc=[]
    for h in H:
        y=ls.shift(-h)-ls; tr=ok&y.notna()
        p=np.sort(model(h).fit(X[tr],y[tr]).predict(xl).reshape(-1))
        fc.append(dict(h=h,date=(last+pd.offsets.BDay(h)).date().isoformat(),q05=s.iloc[-1]*np.exp(p[0]),q50=s.iloc[-1]*np.exp(p[1]),q95=s.iloc[-1]*np.exp(p[2])))
    F=pd.DataFrame(fc); print((F.set_index('date')[['q05','q50','q95']]/1e6).round(1))
    B.to_pickle(f'bt_{key}.pkl'); F.to_csv(f'fc_{key}.csv',index=False)
    OUT[key]=dict(eval=E.to_dict('records'),last=float(s.iloc[-1]),last_date=str(last.date()),byyr12=byyr.reset_index().to_dict('records'))
json.dump(OUT,open('fc_results.json','w'),indent=1,default=float)
