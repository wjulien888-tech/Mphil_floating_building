#!/usr/bin/env python3
"""Plot a completed M0 scenario suite without changing or rerunning it.

PNG and SVG figures share the same source figure. PDF packaging uses the PNGs
so the delivered PDF and conversation previews contain identical graphics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/m0_s0_s5_20260930_234602_526269'
OUT = ROOT / 'deliverables/m0_visualizations_20261002'
TEAL, ORANGE, RED, BLUE = '#087f82', '#d68629', '#bf4b45', '#456c96'
INK, GREY, LIGHT, PALE = '#253441', '#78848e', '#e4e9ec', '#f4f7f8'
DT = .25


def setup():
    font = Path('/System/Library/Fonts/STHeiti Light.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.edgecolor': LIGHT, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': GREY, 'ytick.color': GREY, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'svg.fonttype': 'path', 'pdf.fonttype': 42})


def frame(ax, title=None, ylabel=None):
    ax.grid(axis='y', color=LIGHT, lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)
    if title: ax.set_title(title, loc='left', pad=16)
    if ylabel: ax.set_ylabel(ylabel, labelpad=10)


def head(fig, number, title, subtitle):
    fig.text(.065, .954, f'M0 / {number:02d}', color=TEAL, fontsize=12)
    fig.text(.065, .914, title, fontsize=21)
    fig.text(.065, .873, subtitle, fontsize=10.7, color=GREY)


def foot(fig, lines):
    source='来源：M0 2026-09-30批次；冻结GRU L96/H96；电网充电开启；CL1+CL2为关键负荷。'
    fig.text(.065, .022, '\n'.join(lines+[source]), fontsize=8.7, color=GREY, va='bottom', linespacing=1.6)


def card(fig, x, y, title, value, note='', color=INK, width=.215):
    ax = fig.add_axes([x, y, width, .105]); ax.axis('off')
    ax.axhline(.98, color=LIGHT, lw=1)
    ax.text(0, .73, title, fontsize=10, color=GREY)
    ax.text(0, .32, value, fontsize=22 if len(value)<=9 else 19.5, color=color)
    if note: ax.text(0, .02, note, fontsize=9, color=GREY)


def save(fig, name, out):
    fig.savefig(out / f'{name}.png', dpi=240)
    fig.savefig(out / f'{name}.svg')
    plt.close(fig)


def load(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    state = json.loads((run / 'suite_status.json').read_text())
    s = pd.read_csv(run / 'summary.csv')
    assert state['full_matrix_completed'] and s.status.eq('completed').all()
    assert len(s) == 47 and s.completed_steps.eq(2880).all()
    cases = {c['name']: c for c in manifest['all_cases']}
    for col in ['event_hour', 'outage_hours', 'initial_soc', 'rho', 'pv_regime']:
        s[col] = s['case'].map(lambda name: cases[name][col])
    s['critical_ens_kwh'] = s.outage_cl1_ens_kwh + s.outage_cl2_ens_kwh
    s['critical_load_kwh'] = s.outage_cl1_load_kwh + s.outage_cl2_load_kwh
    s['critical_supply_pct'] = 100 * (1-s.critical_ens_kwh/s.critical_load_kwh)
    s['outage_soc_pct'] = 100 * s.outage_start_soc_kwh / 215
    s['event_timestamp'] = pd.Timestamp(manifest['identity']['config']['start']) + pd.to_timedelta(s.event_hour, unit='h')
    return manifest, cases, s.set_index('case', drop=False)


def actual(s, name):
    return pd.read_csv(Path(s.loc[name, 'result_path']) / 'actual.csv', parse_dates=['timestamp'])


def snapshot(s, name, step):
    for block in pd.read_csv(Path(s.loc[name, 'result_path'])/'plans.csv.gz', chunksize=9600):
        found = block[block.decision_step.eq(step)]
        if len(found):
            assert len(found) == 96
            return found.reset_index(drop=True)
    raise ValueError(f'Missing decision {step} for {name}')


def forecast_data(s):
    a = actual(s, 'S0_grid')
    p = pd.read_csv(Path(s.loc['S0_grid','result_path'])/'plans.csv.gz',
        usecols=['decision_step','decision_timestamp','horizon_timestamp','pv_forecast_kw'])
    truth = a.set_index(a.timestamp.astype(str)).actual_pv_kw
    p['actual_pv_kw'] = p.horizon_timestamp.map(truth)
    g = p.groupby('decision_step')
    r = pd.DataFrame({'timestamp':g.decision_timestamp.first(),
        'forecast_kwh':g.pv_forecast_kw.sum()*DT,
        'actual_kwh':g.actual_pv_kw.sum(min_count=96)*DT,
        'actual_count':g.actual_pv_kw.count()}).reset_index()
    r['timestamp'] = pd.to_datetime(r.timestamp)
    assert len(p) == 2880*96 and (g.size() == 96).all()
    assert r.actual_count.eq(96).sum() == 2785
    return p, r


def selection_plot(s, run, out):
    """Display actual PV and independently verify the selected low-PV windows."""
    a=actual(s,'S0_grid').set_index('timestamp')
    records={e['case']:e for e in json.loads((run/'event_selection.json').read_text())}
    fig=plt.figure(figsize=(13.5,12.5))
    head(fig,1,'低光伏事件如何选出：实际出力、起止区间与筛选依据',
         'S4六个停电案例 / 在2023年11月内，对每种时长单独搜索累计PV发电量最低的连续窗口')
    fig.text(.065,.836,'筛选规则：每天12:00起算；至少24h前置运行和24h恢复期；累计电量 = 每15分钟PV功率 × 0.25h 后求和。',fontsize=10)
    ax=fig.add_axes([.08,.671,.855,.125]);frame(ax,'A  本月实际PV出力；浅色区域覆盖下方全部选中窗口','PV功率（kW）')
    ax.plot(a.index,a.actual_pv_kw,color=TEAL,lw=.85)
    ax.axvspan(pd.Timestamp('2023-11-06 12:00'),pd.Timestamp('2023-11-13 12:00'),color=ORANGE,alpha=.12)
    ax.set_xlim(a.index.min(),a.index.max());ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,5,9,13,17,21,25,29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'));ax.tick_params(labelsize=9)
    fig.text(.08,.616,'B  选中窗口的实际PV曲线（每个小图对应一个S4停电案例）',fontsize=12)
    output=[]
    for j,h in enumerate([6,12,24,48,72,168]):
        name=f'S4_observed_low_outage_{h}h';e=records[name]
        start,end=pd.Timestamp(e['start']),pd.Timestamp(e['end_exclusive'])
        n=int(h/DT);candidates=[]
        for i,t in enumerate(a.index):
            if t.hour!=12 or t.minute!=0:continue
            if i<96 or i+n+96>len(a):continue
            candidates.append((float(a.actual_pv_kw.iloc[i:i+n].sum()*DT),t))
        best,chosen=min(candidates)
        assert chosen==start and len(candidates)==e['candidate_count']
        assert abs(best-e['pv_kwh'])<1e-8
        median=float(np.median([v for v,_ in candidates]));segment=a.loc[(a.index>=start)&(a.index<end)]
        col,row=j%2,j//2;left=.08+col*.475;bottom=.456-row*.167
        bx=fig.add_axes([left,bottom,.38,.102]);frame(bx,ylabel='PV（kW）')
        bx.set_title(f'S4-{h}h  |  {start:%m/%d %H:%M} 至 {end:%m/%d %H:%M}',loc='left',pad=22,fontsize=10.5)
        bx.text(0,1.07,f'{best:.2f} kWh / {len(candidates)}个候选中最低；候选中位数 {median:.2f} kWh',
                transform=bx.transAxes,fontsize=8.5,color=ORANGE)
        xx=segment.index.append(pd.DatetimeIndex([end]));yy=np.r_[segment.actual_pv_kw,segment.actual_pv_kw.iloc[-1]]
        bx.step(xx,yy,where='post',color=TEAL,lw=1.15);bx.fill_between(xx,0,yy,step='post',color=TEAL,alpha=.1)
        bx.set_xlim(start,end);bx.set_ylim(0,max(float(segment.actual_pv_kw.max())*1.18,.5))
        bx.set_xticks([start,start+(end-start)/2,end]);bx.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        bx.tick_params(labelsize=8);bx.yaxis.set_major_locator(plt.MaxNLocator(3))
        output.append(dict(case=name,start=str(start),end_exclusive=str(end),hours=h,pv_kwh=best,
            candidate_count=len(candidates),candidate_median_kwh=median,rank=1,peak_pv_kw=float(segment.actual_pv_kw.max()),
            repaired_pv_points=e['repaired_pv_points']))
    foot(fig,['结束时刻不含；小图纵轴按各自出力标度。PV未统一缩放，时间顺序未重排。最低窗口仅指本月符合规则的候选集合。',
              '含因果hold-last修补点的窗口参与筛选；6/12/24/48/72/168h窗口分别含1/1/0/1/1/2个修补点。'])
    save(fig,'01_低光伏事件选取依据',out)
    pd.DataFrame(output).to_csv(out/'data/S4事件选取与候选核对.csv',index=False)


def s4_results(s,out):
    names=[f'S4_observed_low_outage_{h}h' for h in [6,12,24,48,72,168]]
    d=s.loc[names];x=np.arange(6)
    fig=plt.figure(figsize=(13.5,9.5));head(fig,2,'S4六个独立停电案例：每根柱只统计自己的停电区间',
        '不是六个场景的总和，也不是整月缺供；柱子的颜色表示负荷等级，柱顶数字为该案例三个等级的合计')
    fig.text(.065,.813,'共同设置：月初SOC 50%，备电系数ρ=1，提前24h准确预警，215 kWh / 50 kW电池。事件选取见第1页。',fontsize=10.5)
    ax=fig.add_axes([.09,.415,.845,.31]);frame(ax,'停电期间各级负荷缺供电量','该案例停电期间缺供（kWh）')
    bottom=np.zeros(6)
    for col,label,color in [('outage_cl1_ens_kwh','CL1 安全关键',RED),('outage_cl2_ens_kwh','CL2 功能保障',ORANGE),('outage_cl3_ens_kwh','CL3 可中断',GREY)]:
        ax.bar(x,d[col],bottom=bottom,width=.38,color=color,label=label)
        for i,y in enumerate(d[col]):
            if y>=4:ax.text(i,bottom[i]+y/2,f'{y:.2f}',ha='center',va='center',fontsize=9,color='white')
        bottom+=d[col].to_numpy()
    for i,v in enumerate(bottom):ax.text(i,v+2,f'合计 {v:.2f}',ha='center',fontsize=10,color=INK)
    ax.set_ylim(0,max(bottom)*1.28);ax.set_xlim(-.65,5.65)
    ax.legend(frameon=False,ncol=3,fontsize=10,loc='upper left')
    ax.set_xticks(x,[f'S4-{int(v)}h' for v in d.outage_hours],fontsize=11)
    tx=fig.add_axes([.09,.135,.845,.20]);tx.axis('off')
    rows=['案例','停电开始','停电结束（不含）','关键缺供 CL1+CL2','关键负荷供电率']
    values=[[f'S4-{int(row.outage_hours)}h',f'{row.event_timestamp:%m/%d %H:%M}',
            f'{row.event_timestamp+pd.Timedelta(hours=row.outage_hours):%m/%d %H:%M}',
            f'{row.critical_ens_kwh:.2f} kWh',f'{row.critical_supply_pct:.2f}%'] for _,row in d.iterrows()]
    tab=tx.table(cellText=np.array(values,dtype=object).T.tolist(),rowLabels=rows,loc='center',cellLoc='center',colWidths=[.145]*6)
    tab.auto_set_font_size(False);tab.set_fontsize(9.7);tab.scale(1,1.7)
    for (r,j),cell in tab.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor(PALE if r%2==0 else 'white');cell.set_text_props(color=GREY if j==-1 else INK)
    foot(fig,['范围：仅S4的6个停电案例，不含S4并网对照。S4-Xh对应原始案例名S4_observed_low_outage_Xh。',
              '不同案例的日期、时长和负荷总量不同；原始逐案数值均保留，不把各柱相加作为共同场景的结果。'])
    save(fig,'02_低光伏场景S4',out)


def s5_plot(s,out):
    fig=plt.figure(figsize=(13.5,9.5));head(fig,3,'S5低光伏对照：同一个事件，多备电后少缺多少电？',
        '每组只比较两种备电系数；日期、停电时长和PV相同。柱子越短，关键负荷缺供越少。')
    fig.text(.065,.81,'备电系数ρ：计算备用需求时，先用它乘未来总负荷，再扣除预测PV。这里比较0.5和1；不是实际供电比例。',fontsize=10.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ORANGE,label='较低备电系数 ρ=0.5'),Patch(color=TEAL,label='较高备电系数 ρ=1')],
        frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.52,.765),fontsize=12)
    result=[]
    for j,duration in enumerate([24,168]):
        ax=fig.add_axes([.17+j*.46,.38,.32,.27])
        frame(ax,f'{"A" if j==0 else "B"}  停电{duration}小时')
        names=[f'S5_soc0p2_rho0p5_t{h}_D{duration}_observed_low' for h in [30,42]]
        labels=[]
        for i,name in enumerate(names):
            lo=s.loc[name];hi=s.loc[name.replace('rho0p5','rho1')]
            for row in [lo,hi]:
                other=row['case'].replace('soc0p2','soc0p8')
                assert abs(row.critical_ens_kwh-s.loc[other,'critical_ens_kwh'])<1e-7
            labels.append(f'{lo.event_timestamp:%m/%d %H:%M}\n开始停电')
            for y,v,col in [(i-.14,lo.critical_ens_kwh,ORANGE),(i+.14,hi.critical_ens_kwh,TEAL)]:
                ax.barh(y,v,height=.20,color=col)
                ax.text(v+(0.15 if duration==24 else 1),y,f'{v:.2f}',va='center',fontsize=11,color=col)
            result.append(dict(case_low=name,case_high=hi['case'],start=str(lo.event_timestamp),hours=duration,
                rho05_critical_ens_kwh=lo.critical_ens_kwh,rho1_critical_ens_kwh=hi.critical_ens_kwh,
                reduction_pct=100*(1-hi.critical_ens_kwh/lo.critical_ens_kwh)))
        ax.set_yticks([0,1],labels,fontsize=10);ax.set_ylim(1.48,-.48)
        ax.set_xlim(0,11 if duration==24 else 68);ax.set_xlabel('关键负荷缺供（kWh，越少越好）',labelpad=12)
        ax.grid(False);ax.grid(axis='x',color=LIGHT,lw=.7)
    for x0,r in zip([.065,.535],[result[0],result[2]]):
        box=fig.add_axes([x0,.16,.40,.11]);box.axis('off');box.axhline(.99,color=LIGHT,lw=1)
        box.text(0,.7,f'{r["hours"]}h早晨开始的事件',fontsize=11,color=GREY)
        box.text(0,.30,f'{r["rho05_critical_ens_kwh"]:.2f} → {r["rho1_critical_ens_kwh"]:.2f} kWh',fontsize=23,color=TEAL)
        box.text(0,.01,f'关键缺供减少 {r["reduction_pct"]:.1f}%',fontsize=11,color=TEAL)
    foot(fig,['本图只画S5低光伏16例：4个事件 × 2个备电系数 × 2个月初SOC。两档月初SOC结果重合，每根柱代表2例。',
              '其余16例S5原日期结果保留在完整数据表中。两面板横轴都从0开始，但量程不同；比较同一组内的两根柱。'])
    save(fig,'03_S5备电系数与SOC对照',out)
    pd.DataFrame(result).to_csv(out/'data/S5低光伏配对对照.csv',index=False)


def case_metrics(s,name):
    a=actual(s,name);c=s.loc[name];step=int(c.event_hour*4);n=int(c.outage_hours*4)
    p=snapshot(s,name,step);o=a.iloc[step:step+n]
    ens=(o.cl1_ens_kwh+o.cl2_ens_kwh).sum()
    planned=(o.cl1_scheduled_shed_kw+o.cl2_scheduled_shed_kw).sum()*DT
    usable=(a.iloc[step].soc_before_kwh-21.5)*np.sqrt(.92)
    m=dict(case=name,step=step,event_timestamp=str(a.iloc[step].timestamp),
        first24_forecast_kwh=float(p.pv_forecast_kw.sum()*DT),
        first24_actual_kwh=float(a.iloc[step:step+96].pv_available_kwh.sum()),
        start_soc_pct=float(c.outage_soc_pct),start_usable_battery_kwh=float(usable),
        critical_load_kwh=float(o.critical_load_kwh.sum()),actual_pv_kwh=float(o.pv_available_kwh.sum()),
        critical_ens_kwh=float(ens),planned_critical_shed_kwh=float(planned),additional_critical_ens_kwh=float(ens-planned),
        supply_pct=float(c.critical_supply_pct),energy_gap_lower_bound_kwh=float(max(0,o.critical_load_kwh.sum()-o.pv_available_kwh.sum()-usable)))
    return a,p,m


def island_axis(ax,end,xticks):
    ax.axvline(0,color=GREY,lw=.8,ls='--')
    ax.set_xlim(min(xticks),end);ax.set_xticks(xticks)
    ax.set_axisbelow(True);ax.grid(axis='y',color=LIGHT,lw=.6);ax.tick_params(length=0)


def chain24(s,out,metrics):
    n='S5_soc0p2_rho0p5_t30_D24_observed_low';n1=n.replace('rho0p5','rho1')
    a,p,m=case_metrics(s,n);b,p1,m1=case_metrics(s,n1);metrics.extend([m,m1])
    step=m['step'];sl=slice(step-48,step+96);x=np.arange(-48,96)*DT
    fig=plt.figure(figsize=(13.5,11));head(fig,4,'24小时案例：PV预测为实际的11.1倍，停电时备电偏低',
        'S5 / 11月11日06:00开始停电24h；月初SOC 20%；对照ρ=0.5与ρ=1，使用同一PV预测')
    for x0,title,value,note,color in [(.065,'停电起点预测未来24h PV',f'{m["first24_forecast_kwh"]:.2f} kWh','固定当时的96步预测',ORANGE),
        (.295,'同一24h实际PV',f'{m["first24_actual_kwh"]:.2f} kWh','预测 / 实际 = 11.07',TEAL),
        (.525,'ρ=0.5 停电起始SOC',f'{m["start_soc_pct"]:.2f}%','可放电能量仅7.55 kWh',RED),
        (.755,'ρ=0.5 关键负荷供电率',f'{m["supply_pct"]:.2f}%','关键缺供8.91 kWh',RED)]:card(fig,x0,.74,title,value,note,color)
    ax=fig.add_axes([.085,.555,.85,.13]);bx=fig.add_axes([.085,.347,.85,.13]);cx=fig.add_axes([.085,.14,.85,.13])
    frame(ax,'A  停电开始时，对未来一天的PV判断过于乐观','PV功率（kW）')
    ax.plot(x,a.iloc[sl].actual_pv_kw,color=TEAL,lw=1.8,label='实际PV')
    ax.plot(np.arange(96)*DT,p.pv_forecast_kw,color=ORANGE,lw=1.7,ls='--',label='停电起点的24h预测（固定）')
    ax.fill_between(np.arange(96)*DT,a.iloc[step:step+96].actual_pv_kw,p.pv_forecast_kw,
        where=p.pv_forecast_kw.to_numpy()>a.iloc[step:step+96].actual_pv_kw.to_numpy(),color=ORANGE,alpha=.13)
    ax.legend(frameon=False,fontsize=9,loc='upper left',ncol=2)
    frame(bx,'B  电池随后降至最低SOC；当天实际轨迹远低于原计划','SOC（%）')
    bx.plot(x,a.iloc[sl].soc_before_fraction*100,color=RED,lw=1.9,label='实际SOC：ρ=0.5')
    bx.plot(x,b.iloc[sl].soc_before_fraction*100,color=TEAL,lw=1.6,label='实际SOC：ρ=1')
    bx.plot(np.arange(97)*DT,np.r_[a.iloc[step].soc_before_fraction*100,p.soc_end.to_numpy()/215*100],
        color=GREY,ls='--',lw=1.3,label='ρ=0.5停电起点的SOC计划（固定）')
    bx.axhline(10,color=INK,ls=':',lw=.8);bx.set_ylim(0,95);bx.set_yticks([10,30,50,70,90])
    bx.legend(frameon=False,fontsize=8.6,loc='upper left',ncol=3)
    frame(cx,'C  关键负荷缺供累积；提高备电系数可减轻损失','累计缺供（kWh）')
    for ar,col,label,mm in [(a,RED,'ρ=0.5',m),(b,TEAL,'ρ=1',m1)]:
        ens=(ar.iloc[step:step+96].cl1_ens_kwh+ar.iloc[step:step+96].cl2_ens_kwh).cumsum()
        cx.plot(np.arange(97)*DT,np.r_[0,ens],lw=2,color=col,label=label)
        cx.text(24.35,float(ens.iloc[-1]),f'{mm["critical_ens_kwh"]:.2f}',color=col,va='center',fontsize=11)
    cx.legend(frameon=False,fontsize=9,loc='upper left',ncol=2);cx.set_ylim(0,10.5)
    for ar in [ax,bx,cx]:
        island_axis(ar,25,[-12,-6,0,6,12,18,24]);ar.axvspan(-12,0,color=PALE,zorder=-2)
    ax.set_xticklabels([]);bx.set_xticklabels([]);cx.set_xlabel('相对停电开始时间（小时）；灰色区间为提前12h预警期')
    foot(fig,[f'ρ=0.5时，24h关键需求27.86 kWh，实际PV13.55 kWh，起点电池可放电7.55 kWh；即便PV全给关键负荷，总量仍至少差{m["energy_gap_lower_bound_kwh"]:.2f} kWh。',
        '该下限固定本次停电起始SOC，并忽略时序约束；曲线展示已发生的预测、计划与执行，不等于已完成Perfect-PV因果对照。'])
    save(fig,'04_24h预测高估与备电缺供证据链',out)


def chain72(s,out,metrics):
    name='S4_observed_low_outage_72h';a,p,m=case_metrics(s,name);metrics.append(m)
    step=m['step'];o=a.iloc[step:step+288];x=np.arange(288)*DT
    fig=plt.figure(figsize=(13.5,11));head(fig,5,'72小时案例：首日PV高估后，后续两天持续出现缺供',
        'S4 / 11月7日12:00开始停电72h；ρ=1；提前24h预警；连续低发电窗口来自源数据')
    card(fig,.065,.74,'起点预测前24h PV',f'{m["first24_forecast_kwh"]:.2f} kWh','同一24h实际只有24.62 kWh',ORANGE)
    card(fig,.295,.74,'停电起始SOC',f'{m["start_soc_pct"]:.2f}%','215 kWh电池，SOC下限10%')
    card(fig,.525,.74,'72h关键负荷缺供',f'{m["critical_ens_kwh"]:.2f} kWh','关键负荷供电率79.13%',RED)
    card(fig,.755,.74,'其中计划切负荷',f'{m["planned_critical_shed_kwh"]:.2f} kWh','执行时额外缺供5.49 kWh',ORANGE)
    ax=fig.add_axes([.085,.555,.85,.13]);bx=fig.add_axes([.085,.347,.85,.13]);cx=fig.add_axes([.085,.14,.85,.13])
    frame(ax,'A  预测高估首日发电；实际低发电持续三天','PV功率（kW）')
    ax.plot(x,o.actual_pv_kw,color=TEAL,lw=1.7,label='实际PV')
    ax.plot(np.arange(96)*DT,p.pv_forecast_kw,color=ORANGE,lw=1.7,ls='--',label='停电起点的前24h预测（固定）')
    ax.legend(frameon=False,fontsize=9,loc='upper right',ncol=2)
    frame(bx,'B  实际SOC低于滚动备电目标，备用缺口在执行中显现','SOC（%）')
    bx.plot(x,o.soc_before_fraction*100,color=RED,lw=1.8,label='实际SOC')
    bx.plot(x,o.reserve_target_kwh/215*100,color=ORANGE,ls='--',lw=1.3,label='每轮更新的备电目标')
    bx.plot(np.arange(97)*DT,np.r_[m['start_soc_pct'],p.soc_end/215*100],color=GREY,ls=':',lw=1.4,label='停电起点的前24h SOC计划')
    bx.axhline(10,color=INK,ls=':',lw=.8);bx.set_ylim(0,95);bx.set_yticks([10,30,50,70,90])
    bx.legend(frameon=False,fontsize=8.6,loc='upper right',ncol=3)
    frame(cx,'C  已计划的切负荷与执行时额外缺供共同累积','累计关键缺供（kWh）')
    ens=(o.cl1_ens_kwh+o.cl2_ens_kwh).to_numpy()
    scheduled=(o.cl1_scheduled_shed_kw+o.cl2_scheduled_shed_kw).to_numpy()*DT
    extra=ens-scheduled;assert extra.min()>-1e-7
    xx=np.arange(289)*DT;cp=np.r_[0,scheduled.cumsum()];ce=np.r_[0,np.maximum(extra,0).cumsum()]
    cx.fill_between(xx,0,cp,color=ORANGE,alpha=.72,label='MPC计划切除的关键负荷')
    cx.fill_between(xx,cp,cp+ce,color=RED,alpha=.8,label='执行时额外缺供')
    cx.plot(xx,cp+ce,color=RED,lw=1.5);cx.text(72.5,float(ens.sum()),f'{ens.sum():.2f}',color=RED,va='center')
    cx.legend(frameon=False,fontsize=9,loc='upper left',ncol=2);cx.set_ylim(0,38)
    for ar in [ax,bx,cx]:
        island_axis(ar,75,[0,12,24,36,48,60,72])
        for h in [24,48]:ar.axvline(h,color=LIGHT,ls='--',lw=.8)
    ax.set_xticklabels([]);bx.set_xticklabels([]);cx.set_xlabel('相对停电开始时间（小时）')
    foot(fig,['起点可放电75.61 kWh + 72h实际PV107.85 kWh > 关键需求149.69 kWh；本例不能仅凭缺供判定储能容量不足。',
              '计划切负荷按各轮实际执行的第一步累加；备用目标超过24h的部分重复预测日估算，仍需检查长时能源分配与执行规则。'])
    save(fig,'05_72h持续低发电与关键负荷缺供',out)


def rolling_plot(s,p,r,out):
    valid=r[r.actual_count.eq(96)].copy()
    steps=np.sort((s.loc[s.outage_hours.gt(0),'event_hour'].unique()*4).astype(int))
    e=r.set_index('decision_step').loc[steps].copy();e['ratio']=e.forecast_kwh/e.actual_kwh
    e['error_kwh']=e.forecast_kwh-e.actual_kwh
    fig=plt.figure(figsize=(13.5,9.5));head(fig,6,'回看整月：未来24h发电量预测在低发电事件中偏高',
        '全部47例共享相同PV输入和预测；每个点都是当时输出的完整96步预测，不是滚动第一步预测的累加')
    ax=fig.add_axes([.085,.55,.85,.23]);frame(ax,'A  本月所有完整24h窗口：预测总发电量与实际值','未来24h电量（kWh）')
    ax.plot(valid.timestamp,valid.forecast_kwh,color=ORANGE,lw=1.25,label='M0预测')
    ax.plot(valid.timestamp,valid.actual_kwh,color=TEAL,lw=1.6,label='对应窗口实际PV')
    ax.fill_between(valid.timestamp,valid.actual_kwh,valid.forecast_kwh,
        where=valid.forecast_kwh>valid.actual_kwh,color=ORANGE,alpha=.11)
    ax.set_xlim(valid.timestamp.min(),valid.timestamp.max())
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,5,9,13,17,21,25,29]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.legend(frameon=False,loc='upper right',ncol=2,fontsize=9);ax.set_xlabel('做出预测的时刻')
    bx=fig.add_axes([.14,.15,.44,.27]);frame(bx,'B  所有10个不同停电起点的24h预测','PV电量（kWh）')
    x=np.arange(len(e));bx.vlines(x,e.actual_kwh,e.forecast_kwh,color=LIGHT,lw=3)
    bx.scatter(x,e.forecast_kwh,color=ORANGE,s=38,label='预测',zorder=3)
    bx.scatter(x,e.actual_kwh,color=TEAL,s=38,label='实际',zorder=3)
    bx.set_xticks(x,[t.strftime('%m/%d\n%H:%M') for t in e.timestamp],fontsize=8);bx.legend(frameon=False,ncol=2,fontsize=8)
    tx=fig.add_axes([.67,.16,.27,.26]);tx.axis('off')
    largest=e.loc[984];event_count=int((e.error_kwh>0).sum())
    tx.text(0,.93,f'{event_count}/10 个停电起点高估未来24h PV',fontsize=13,color=ORANGE)
    tx.text(0,.7,'11/11 06:00：150.06 vs 13.55 kWh\n11/07 12:00：90.19 vs 24.62 kWh',fontsize=11,linespacing=1.8)
    tx.text(0,.30,'事件起点合并了重复的时长、ρ和月初SOC。\n高估可能降低预期缺口与备电需求。\n损失大小还取决于SOC、ρ及执行规则。',fontsize=10,color=GREY,linespacing=1.8)
    foot(fig,['上图含2,785个完整实际窗口；末尾95个窗口超出本月执行记录，未补造实际数据，未纳入上图。',
              '滚动窗口相互重叠；10个事件是本轮固定测试设计，不是独立随机样本，也不代表全年高估概率。'])
    save(fig,'06_整月24h预测与全部事件核对',out)
    return e.reset_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();setup();out=args.output;out.mkdir(parents=True,exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
    manifest,cases,s=load(args.run);p,r=forecast_data(s);metrics=[]
    selection_plot(s,args.run,out);s4_results(s,out);s5_plot(s,out)
    chain24(s,out,metrics);chain72(s,out,metrics);events=rolling_plot(s,p,r,out)
    s.to_csv(data/'全部47案例绘图指标.csv',index=False)
    r.to_csv(data/'整月滚动24h预测电量.csv',index=False)
    events.to_csv(data/'全部10个停电起点预测.csv',index=False)
    pd.DataFrame(metrics).to_csv(data/'重点案例证据链.csv',index=False)
    record={'source_run':str(args.run.resolve()),'source_manifest_sha256':hashlib.sha256((args.run/'manifest.json').read_bytes()).hexdigest(),
        'source_summary_sha256':hashlib.sha256((args.run/'summary.csv').read_bytes()).hexdigest(),
        'case_count':47,'steps':135360,'critical_definition':'CL1 + CL2',
        'model':'frozen GRU, L96 H96, seed42','allow_grid_charging':manifest['allow_grid_charging'],
        'full_actual_24h_windows':int(r.actual_count.eq(96).sum()),'outage_start_instants':len(events),
        'case_selection':'24h: lowest critical supply in S5, paired rho1; 72h: lowest critical supply in S4',
        'numerical_tolerance':'Tiny solver-level unmet energy is retained in data; described as non-material below 1e-4 kWh.',
        'invalid_pv_policy':manifest['identity']['invalid_pv'],
        'source_repairs':['2023-11-08 17:45:00','2023-11-11 17:45:00','2023-11-30 17:45:00'],
        'causal_boundary':'Observed forecast-plan-execution sequence, not a perfect-forecast counterfactual. No simulator modification or rerun.',
        'figures':[p.name for p in sorted(out.glob('*.png'))]}
    (data/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (out/'图表说明.txt').write_text('M0本轮实验结果图集\n\n'+
        '数据：m0_s0_s5_20260930_234602_526269；47例全部完成；2023年11月，每例30天。\n'+
        '图01：S4低光伏事件的选取依据，整月PV及六个实际窗口出力曲线。\n图02：S4六个独立停电案例，各自停电期间的分级缺供，明确逐案统计范围。\n'+
        '图03：S5低光伏16案例，按24h和168h分开，对比同一事件的两档备电系数。月初SOC两档结果重合，每根柱对应2例。其余S5原日期16例保留完整数据。\n'+
        '图04：S5最低关键负荷供电率案例，与同条件rho=1对照。固定预测是停电开始时的一次96步输出。\n'+
        '图05：S4最低关键负荷供电率案例，分开累计计划切负荷和执行时额外缺供。\n'+
        '图06：整月滚动24h电量预测及全部10个不同停电起点，展示案例之外的预测表现。\n\n'+
        '关键负荷=CL1+CL2；供电率按停电期间实际供电量/需求量计算。计划切负荷不是实际PV偏差造成的额外缺供。\n'+
        '低PV来自连续源数据窗口，不使用人为统一缩放。不同窗口起点、负荷和运行历史不同。\n'+
        '本轮3个无效PV点依既定hold-last规则用此前有效值修补；重点24h和72h窗口各含1个修补点。\n'+
        '保留本轮准确停电预警、已知未来负荷、长时备电重复预测日等实验设置。\n'+
        '图04总能量缺口下限固定实际停电SOC，并将所有PV优先给关键负荷、忽略时序约束，故不是重新求解的反事实结果。\n'+
        '图05全时段总能量不构成关键负荷缺供的充分解释；控制计划和能量到达时间同样重要。\n'+
        'PNG为240dpi；SVG为矢量路径。PDF按同一PNG逐页封装，以保持预览和PDF内容一致。\n'+
        '原日期场景小差异图按批注删除，S0–S3及全部S5的数值仍保留在完整47案例数据表中。各图可复算指标见data目录；未修改或重跑实验。\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
