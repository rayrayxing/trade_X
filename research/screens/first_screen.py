import pandas as pd, numpy as np, json
# Inputs (real data): datasets/exchange-rates data/daily.csv -> data/fx.csv,
# datasets/finance-vix data/vix-daily.csv -> data/vix.csv, datasets/s-and-p-500 data/data.csv -> data/spx.csv
D="data/"
fx=pd.read_csv(D+"fx.csv"); fx.columns=["date","country","v"]; fx["date"]=pd.to_datetime(fx.date)
fx=fx.pivot_table(index="date",columns="country",values="v").replace(0,np.nan)
usd_per={"Euro":"EUR","United Kingdom":"GBP","Australia":"AUD","New Zealand":"NZD"}
inv={"Japan":"JPY","Canada":"CAD","Switzerland":"CHF","Sweden":"SEK","Norway":"NOK"}
px=pd.DataFrame({c:fx[k] for k,c in usd_per.items()})
for k,c in inv.items(): px[c]=1/fx[k]
px=px.loc["2000-01-01":].ffill(limit=3).dropna()
ret=px.pct_change().fillna(0)
COST_FX=2e-4
def stats(r,name,per=252):
    r=r.dropna(); out={}
    for lab,rr in (("full",r),("2016+",r.loc["2016-01-01":])):
        if len(rr)<10: continue
        mu=rr.mean()*per; sd=rr.std()*np.sqrt(per); sh=mu/sd if sd>0 else np.nan
        eq=(1+rr).cumprod(); dd=(eq/eq.cummax()-1).min()
        out[lab]=dict(ann_ret=round(mu*100,1),ann_vol=round(sd*100,1),sharpe=round(sh,2),
                      t=round(sh*np.sqrt(len(rr)/per),1),maxdd=round(dd*100,1))
    return out
def run_ts(pos):
    pos=pos.shift(1).fillna(0)  # trade on next day
    gross=(pos*ret).mean(axis=1)
    turn=pos.diff().abs().fillna(0).mean(axis=1)
    return gross-turn*COST_FX
vol=ret.rolling(60).std()*np.sqrt(252)
scale=(0.10/vol).clip(upper=3)
res={}
def weekly(p): return p.resample("W-FRI").last().reindex(p.index).ffill()
res["FX TSMOM 12m vol-scaled"]=stats(run_ts(weekly(np.sign(px.pct_change(252))*scale)),"")
res["FX TSMOM 3m vol-scaled"]=stats(run_ts(weekly(np.sign(px.pct_change(63))*scale)),"")
ma=np.sign(px.rolling(50).mean()-px.rolling(200).mean())
res["FX MA 50/200 trend"]=stats(run_ts(ma*scale),"")
hi55=px.rolling(55).max().shift(1); lo55=px.rolling(55).min().shift(1); hi20=px.rolling(20).max().shift(1); lo20=px.rolling(20).min().shift(1)
pos=pd.DataFrame(0.0,index=px.index,columns=px.columns)
for c in px:
    p=0.0; arr=[]
    for t in range(len(px)):
        x=px[c].iat[t]
        if p==0:
            if x>hi55[c].iat[t]: p=1
            elif x<lo55[c].iat[t]: p=-1
        elif p==1 and x<lo20[c].iat[t]: p=0
        elif p==-1 and x>hi20[c].iat[t]: p=0
        arr.append(p)
    pos[c]=arr
res["FX Donchian 55/20 breakout"]=stats(run_ts(pos*scale),"")
res["FX 5-day reversal"]=stats(run_ts(-np.sign(px.pct_change(5))*scale),"")
m=px.resample("ME").last(); mom=m.pct_change(3)
rk=mom.rank(axis=1); n=m.shape[1]
w=((rk>n-3).astype(float)-(rk<=3).astype(float))/3
wd=w.reindex(px.index).ffill().fillna(0)
res["FX cross-sectional momentum 3m"]=stats(run_ts(wd),"")
# equities monthly
s=pd.read_csv(D+"spx.csv",parse_dates=["Date"]).set_index("Date")["SP500"]
s=s.loc["1989-01-01":]; sr=s.pct_change()
vix=pd.read_csv(D+"vix.csv",parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
vm=vix.resample("MS").mean()  # monthly average, aligned with Shiller monthly averages
COST_EQ=5e-4
def run_m(pos):
    pos=pos.shift(2).fillna(0)  # extra month: Shiller prices are monthly averages
    return pos*sr - pos.diff().abs().fillna(0)*COST_EQ
res["S&P buy and hold (monthly)"]=stats(sr.loc["1991":],"",12)
res["S&P 10-month SMA timing"]=stats(run_m((s>s.rolling(10).mean()).astype(float)).loc["1991":],"",12)
res["S&P 12m TSMOM long/flat"]=stats(run_m((s.pct_change(12)>0).astype(float)).loc["1991":],"",12)
vmm=vm.reindex(s.index)
tgt=vmm.median()
res["S&P vol-managed (target/VIX, cap 2x)"]=stats(run_m((tgt/vmm).clip(upper=2)).loc["1991":],"",12)
spike=(vmm>30).astype(float); hold=spike.rolling(3,min_periods=1).max()
res["S&P buy after VIX > 30 (hold 3m)"]=stats(run_m(hold).loc["1991":],"",12)
res["S&P only when VIX < 20"]=stats(run_m((vmm<20).astype(float)).loc["1991":],"",12)
json.dump(res,open("screen_results.json","w"),indent=1)
for k,v in res.items(): print(k, v)
print("FX range", px.index[0].date(), px.index[-1].date(), "SPX", s.index[-1].date(), "VIX", vix.index[-1].date())
print("months VIX>30:", int(spike.loc['1991':].sum()))
