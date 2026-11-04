"""Raw measured CWB PV exploratory weather-event audit. No imputation."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/cwb_weather_events_20261026'
OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'HKUST_clea_bay_PV+wind_data/Dataset/Time series dataset'
sys.path.insert(0,str(ROOT))
from align_hkust_pv_15min import DEFAULT_SITES
idx=pd.date_range('2021-01-01','2024-01-01',freq='15min',inclusive='left')
pv=pd.DataFrame(index=idx)
for name in DEFAULT_SITES:
    d=pd.read_csv(BASE/'PV generation dataset/PV stations with panel level optimizer/Site level dataset'/name,na_values=['NA'])
    t=pd.to_datetime(d.Time)
    s=pd.Series(pd.to_numeric(d['power(W)'],errors='coerce').values/1000,index=t)
    assert not s.index.duplicated().any()
    pv[name]=s.where(s>=0).reindex(idx)
pv.to_csv(OUT/'raw_eight_sites_15min.csv',index_label='datetime')
day=(idx.hour>=6)&(idx.hour<18)
complete=pv.notna().all(axis=1)
power=pv.sum(axis=1,min_count=8)
daily=pd.DataFrame(index=pd.date_range('2021-01-01','2023-12-31'))
daily['daylight_complete_fraction']=complete[day].resample('D').mean()
daily['pv_06_18_kWh']=(power[day]*.25).resample('D').sum(min_count=48)
daily['pv_24h_kWh']=(power*.25).resample('D').sum(min_count=96)
daily['pv_peak_kW']=power[day].resample('D').max()
# Adjacent 15-min ramps, daylight only; includes cloud changes and solar geometry.
ramp=power.diff().abs().where(day & np.r_[False,day[:-1]])
daily['ramp_p95_kW']=ramp.resample('D').quantile(.95)
daily['ramp_total_kW']=ramp.resample('D').sum(min_count=47)
daily['ramp_per_energy']=daily.ramp_total_kW/daily.pv_06_18_kWh
weather=[]
for year in (2021,2022,2023):
    f=BASE/f'Meteorological dataset/Irradiance/Irradiance_{year}.csv'
    d=pd.read_csv(f); d.Time=pd.to_datetime(d.Time); d=d.set_index('Time').sort_index()
    s=pd.to_numeric(d.iloc[:,0],errors='coerce'); s=s[~s.index.duplicated()].reindex(pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='min',inclusive='left'))
    mask=(s.index.hour>=6)&(s.index.hour<18)
    w=pd.DataFrame({'irradiance_06_18_kWh_m2':s[mask].clip(lower=0).resample('D').sum(min_count=684)/60000,'weather_daylight_coverage':s[mask].notna().resample('D').mean()})
    weather.append(w)
daily=daily.join(pd.concat(weather))
daily['local31d_p80_kWh']=daily.pv_06_18_kWh.rolling(31,center=True,min_periods=15).quantile(.8)
daily['energy_ratio_local_p80']=daily.pv_06_18_kWh/daily.local31d_p80_kWh
daily['energy_change_pct']=daily.pv_06_18_kWh.pct_change(fill_method=None)*100
daily.to_csv(OUT/'daily_metrics.csv',index_label='date')
daily.nsmallest(35,'energy_ratio_local_p80').to_csv(OUT/'lowest_days.csv',index_label='date')
daily.nlargest(35,'ramp_p95_kW').to_csv(OUT/'largest_ramp_days.csv',index_label='date')
low=daily.energy_ratio_local_p80.lt(.35)&daily.pv_06_18_kWh.notna()
groups=low.ne(low.shift()).cumsum(); runs=[]
for _,x in daily[low].groupby(groups[low]):
    if len(x)>=2:runs.append(dict(start=str(x.index[0].date()),end=str(x.index[-1].date()),days=len(x),mean_ratio=x.energy_ratio_local_p80.mean(),pv_mean_kWh=x.pv_06_18_kWh.mean()))
pd.DataFrame(runs).to_csv(OUT/'consecutive_low_runs.csv',index=False)
windows=[('Lionrock_Kompasu','2021-10-05','2021-10-17'),('May_rain_2022','2022-05-08','2022-05-17'),('Ma_on','2022-08-21','2022-08-29'),('Nalgae','2022-10-28','2022-11-07'),('Talim','2023-07-13','2023-07-22'),('Saola_Haikui','2023-08-28','2023-09-15'),('Koinu','2023-10-04','2023-10-13')]
for label,a,b in windows:daily.loc[a:b].to_csv(OUT/f'event_{label}.csv',index_label='date')
print('LOW RUNS',pd.DataFrame(runs).to_string(index=False))
print('LOWEST',daily.nsmallest(12,'energy_ratio_local_p80').round(3).to_string())
print('RAMPS',daily.nlargest(10,'ramp_p95_kW')[['pv_06_18_kWh','ramp_p95_kW','energy_ratio_local_p80']].round(2).to_string())
print('WINDOWS')
for label,a,b in windows: print(label, daily.loc[a:b,['pv_06_18_kWh','energy_ratio_local_p80','daylight_complete_fraction','irradiance_06_18_kWh_m2']].round(3).to_string())
json.dump({'source':'raw measured eight site power(W), not building PVGIS','sites':DEFAULT_SITES,'timezone':'timestamps treated as Hong Kong local clock, UTC+8; raw README does not explicitly declare timezone','energy':'06:00 inclusive to 18:00 exclusive, 48 x 15-min samples; missing any of eight sites at any daylight sample invalidates daily energy; no imputation','baseline':'centered 31-calendar-day local P80, min 15 valid days; retrospective descriptive reference, not forecasting input','low_run':'at least 2 consecutive valid days below 35% of local P80; exploratory threshold, not significance test','weather':'minute irradiance integral 06-18, >=95% coverage; missing minutes not filled; small negative values clipped to zero','ramp':'absolute adjacent 15-minute power differences 06-18, p95, unnormalized across seasons'},open(OUT/'method.json','w'),indent=2)
