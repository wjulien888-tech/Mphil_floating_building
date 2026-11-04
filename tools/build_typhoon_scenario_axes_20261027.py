"""Evidence-bounded initial scenario stratification, not fitted outage probabilities."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/typhoon_scenario_axes_20261027';OUT.mkdir(parents=True,exist_ok=True)
raw=[
(2021,'Koguma','小熊','Tropical storm','06-11 16:15','06-12 14:10',1,None,None),
(2021,'Cempaka','查帕卡','Typhoon','07-18 21:40','07-20 19:40',3,None,None),
(2021,'Lupit','卢碧','Tropical storm','08-02 21:40','08-04 18:20',3,None,None),
(2021,'Lionrock','狮子山','Tropical storm','10-08 04:40','10-10 14:20',8,'10-09 06:40','10-10 04:40'),
(2021,'Kompasu','圆规','Typhoon','10-12 00:40','10-14 06:20',8,'10-12 17:20','10-13 16:40'),
(2021,'Rai','雷伊','Super typhoon','12-20 11:20','12-21 12:20',1,None,None),
(2022,'Chaba','暹芭','Typhoon','06-29 21:10','07-03 19:40',8,'07-01 19:10','07-02 16:20'),
(2022,'Mulan','木兰','Tropical storm','08-09 03:40','08-10 18:20',3,None,None),
(2022,'Ma-on','马鞍','Typhoon','08-23 21:10','08-25 16:10',8,'08-24 19:25','08-25 09:20'),
(2022,'Nesat','纳沙','Typhoon','10-16 21:20','10-18 15:40',3,None,None),
(2022,'Nalgae','尼格','Severe tropical storm','10-30 22:10','11-03 06:20',8,'11-02 13:40','11-03 05:20'),
(2023,'Talim','泰利','Typhoon','07-15 04:40','07-18 08:40',8,'07-17 00:40','07-17 16:20'),
(2023,'Doksuri','杜苏芮','Super typhoon','07-26 20:40','07-28 12:40',1,None,None),
(2023,'Saola','苏拉','Super typhoon','08-30 17:40','09-02 23:40',10,'09-01 02:40','09-02 16:20'),
(2023,'Haikui','海葵','Severe typhoon','09-04 04:40','09-05 21:40',1,None,None),
(2023,'Koinu','小犬','Severe typhoon','10-04 21:40','10-09 16:20',9,'10-08 12:40','10-09 11:40')]
d=pd.read_csv(ROOT/'deliverables/cwb_weather_events_20261026/daily_metrics.csv',index_col=0,parse_dates=True)
rows=[]
for year,name,cn,grade,a,b,sig,c,e in raw:
    start=pd.Timestamp(f'{year}-{a}');end=pd.Timestamp(f'{year}-{b}')
    # Fixed retrospective window: local warning start day through cancellation day + 3 days.
    z=d.loc[start.normalize():end.normalize()+pd.Timedelta(days=3)]
    rows.append(dict(year=year,name=name,name_cn=cn,hko_report_category=grade,category_note='HKO report naming category; not local wind nor CMA landfall category',warning_start_hkt=str(start),warning_end_hkt=str(end),warning_hours=(end-start).total_seconds()/3600,max_hk_signal=sig,signal_ge8_hours=0 if c is None else (pd.Timestamp(f'{year}-{e}')-pd.Timestamp(f'{year}-{c}')).total_seconds()/3600,pv_window_start=str(z.index.min().date()),pv_window_end=str(z.index.max().date()),pv_valid_days=int(z.pv_06_18_kWh.notna().sum()),pv_total_days=len(z),pv_min_ratio=z.energy_ratio_local_p80.min(),pv_mean_ratio=z.energy_ratio_local_p80.mean(),pv_missing_days=int(z.pv_06_18_kWh.isna().sum()),source=f'https://www.hko.gov.hk/en/publica/tc/tc{year}/table43.html'))
df=pd.DataFrame(rows);df.to_csv(OUT/'named_typhoon_catalog_2021_2023.csv',index=False)
losses=[(2021,'Lionrock',34742939),(2021,'Kompasu',939530),(2022,'Chaba',1205275),(2022,'Ma-on',768720),(2022,'Nalgae',606770),(2023,'Talim',756562),(2023,'Saola',69188779),(2023,'Haikui_and_remnant_rainstorm',137622024),(2023,'Koinu',1307735)]
pd.DataFrame([dict(year=y,event=n,reported_noninsured_loss_HKD=v,geography='Hong Kong',scope='Reported government/public utilities/other organizations; insured items excluded; not total economic loss, not power-system damage',source=f'https://www.hko.gov.hk/en/publica/tc/tc{y}/annex1.html') for y,n,v in losses]).to_csv(OUT/'damage_evidence.csv',index=False)
power=[
dict(event='Cempaka_2021',geography='affected customers in CSG report; not GBA-only',affected_customers=125700,restoration_snapshot='2021-07-21 16:00',restored_fraction=.95,source='https://www.cpnn.com.cn/news/gdxw/202107/t20210722_1408222.html',note='No per-customer outage start/end available'),
dict(event='Chaba_2022',geography='Guangdong; includes non-GBA cities',affected_customers=582000,restoration_snapshot='2022-07-04 16:00',restored_customers=555000,source='https://huacheng.gz-cmc.com/pages/2022/07/04/f2fca78d133242bd917aed7d8b2fed03.html',note='Not GBA outage count; no individual outage durations'),
dict(event='Ma_on_2022',geography='CSG service area, not GBA-only',tripped_lines_10kV_and_above=187,restoration_snapshot='2022-08-25 17:00',source='https://china-safety.org.cn/Article/Detail_21848_13_20_4_31.html',note='Line trips are not customer outages'),
dict(event='Saola_2023',geography='Guangdong; includes non-GBA cities',affected_customers=148000,restoration_snapshot='2023-09-02 15:00',restored_fraction_lower_bound=.90,source='https://huacheng.gz-cmc.com/pages/2023/09/02/22a813824c0e436e888112baf228a2f3.html',note='Same source confirms supply corridors to Macau unaffected; local failures in Zhuhai/Huizhou'),
dict(event='Saola_2023',geography='Guangdong',restoration_snapshot='2023-09-03 16:00',restored_fraction=1.0,source='https://finance.sina.com.cn/jjxw/2023-09-04/doc-imzknzyi9842590.shtml',note='CSG report republished from SASAC; full restoration timestamp is not duration for every customer')]
pd.DataFrame(power).to_csv(OUT/'power_impact_evidence.csv',index=False)
# Explicit rule-based stratification of complete daily observations, not unsupervised clustering.
z=d[d.pv_06_18_kWh.notna()].copy();z['pv_group']='P0_reference_or_moderate'
z.loc[z.energy_ratio_local_p80.lt(.35),'pv_group']='P1_low_energy'
z.loc[z.energy_ratio_local_p80.ge(.35)&z.ramp_p95_kW.ge(100),'pv_group']='P2_high_ramp'
z.to_csv(OUT/'pv_daily_initial_groups.csv',index_label='date')
summary={'scope':'2021-2023 named tropical cyclones issuing HK warnings; 16 named events, not exhaustive GBA storm census. Unnamed tropical depressions excluded. Guangdong context evidence separately scoped.', 'counts_by_year':df.groupby('year').size().to_dict(),'local_ge8_counts_by_year':df[df.max_hk_signal.ge(8)].groupby('year').size().to_dict(),'warning_hours_summary':df.warning_hours.describe().to_dict(),'ge8_hours_summary':df.loc[df.max_hk_signal.ge(8),'signal_ge8_hours'].describe().to_dict(),'pv_rule_groups':z.pv_group.value_counts().to_dict(),'rule_note':'P1 ratio <0.35 of centered local31d P80; P2 otherwise 15-min abs-ramp P95 >=100 kW for fixed 8-site fleet; thresholds exploratory and capacity-specific, not learned clusters or significance levels. P0 is remainder, not guaranteed stable/normal. Daily groups are not event-independent frequencies.','grid_groups':{'G0':'grid stays available','G1':'one temporary outage then restoration','G2':'repeated outages or delayed/staged restoration; duration unknown until service-specific evidence'},'outage_duration_sensitivity_hours':[1,6,12,24,48],'duration_note':'Engineering sensitivity grid only, not empirically fitted quantiles. 72/168h reserved for explicit extreme stress tests unless local outage evidence found.','sources_asof':'2026-10-27','leakage_note':'Centered P80 and event grouping retrospective only; train thresholds on training split, hold out entire storms; no use of future event knowledge in operational forecasts.'}
(OUT/'method_and_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(df[['year','name','warning_hours','max_hk_signal','signal_ge8_hours','pv_min_ratio','pv_missing_days']].round(3).to_string(index=False));print(json.dumps(summary,ensure_ascii=False,indent=2))
