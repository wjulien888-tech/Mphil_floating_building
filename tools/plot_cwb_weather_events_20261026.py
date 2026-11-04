from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0,'/Users/julia/.codex/skills/research-figure/.venv/lib/python3.14/site-packages')
import scienceplots
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'deliverables/cwb_weather_events_20261026'
d=pd.read_csv(OUT/'daily_metrics.csv',index_col=0,parse_dates=True)
x=pd.read_csv(OUT/'raw_eight_sites_15min.csv',index_col=0,parse_dates=True)
p=x.sum(axis=1,min_count=8)
windows=[('2021-10-05','2021-10-17','Lionrock / Kompasu'),('2022-03-15','2022-04-03','Fog / cold fronts'),('2022-05-08','2022-05-18','Persistent rain'),('2023-07-13','2023-07-22','Talim'),('2023-08-28','2023-09-17','Saola / Haikui remnant'),('2023-10-04','2023-10-13','Koinu')]
with plt.style.context(['science','no-latex']):
    plt.rcParams.update({'font.size':10,'axes.titlesize':11})
    fig,axs=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for ax,(a,b,title) in zip(axs.flat,windows):
        z=d.loc[a:b]; ax.plot(z.index,z.pv_06_18_kWh,'o-',ms=3,color='#0072B2',label='Measured PV, 06-18 h')
        ax.plot(z.index,z.local31d_p80_kWh,'--',color='#777777',label='Local 31-day P80')
        missing=z.pv_06_18_kWh.isna()
        if missing.any():ax.scatter(z.index[missing],np.full(missing.sum(),30),marker='x',color='#D55E00',label='Incomplete PV day')
        ax.set(title=title,ylabel='Eight-site energy (kWh)',ylim=(0,1900)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.grid(axis='y',alpha=.2)
    axs.flat[0].legend(fontsize=8,loc='upper right')
    axs.flat[4].legend(fontsize=8,loc='upper right')
    fig.suptitle('Clear Water Bay: raw measured PV, 2021-2023\nFixed eight sites; no missing-value imputation; dates use local clock',fontsize=13)
    fig.savefig(OUT/'weather_event_daily.png',dpi=180);fig.savefig(OUT/'weather_event_daily.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for ax,(a,b,title) in zip(axs,[('2023-06-05','2023-06-12','Sunny / shower alternation, June 2023'),('2021-10-06','2021-10-14','Two successive tropical cyclones, October 2021'),('2023-08-30','2023-09-04','Saola: abrupt suppression and recovery, 2023')]):
        ax.plot(p.loc[a:b].index,p.loc[a:b],lw=.9,color='#0072B2');ax.set(title=title,ylabel='Power (kW)',ylim=(0,320));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'weather_event_15min.png',dpi=180);fig.savefig(OUT/'weather_event_15min.pdf');plt.close(fig)
print('June ramps',d.loc['2023-06-05':'2023-06-12',['pv_06_18_kWh','ramp_p95_kW']].round(2).to_string())
for day in ['2023-06-08','2023-06-09','2023-06-12','2023-07-06']:
    q=p.loc[day].between_time('06:00','17:45').diff();print(day,'ramps>50',int(q.abs().gt(50).sum()),'max ramp',q.abs().max(),'peak',p.loc[day].max())
manifest={}
for f in [OUT/'daily_metrics.csv',OUT/'raw_eight_sites_15min.csv']:
    manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
json.dump({'input_sha256':manifest,'style':'SciencePlots science + no-latex','figures':['weather_event_daily.png','weather_event_15min.png'],'visual_check':'pending'},open(OUT/'figure_manifest.json','w'),indent=2)
