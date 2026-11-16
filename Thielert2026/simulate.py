"""Illustrative classical-error simulation; no parameters fitted to the preprint.
Run: python simulate.py (requires numpy and matplotlib).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261116)
n=318
# Values are centered log2 abundances. True protein equals true RNA exactly.
z=rng.normal(size=n); er=rng.normal(size=n); ep=rng.normal(size=n)
tech=.20
sigmas=[1.0,.20,.04]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#152936','text.color':'#152936','axes.titleweight':'bold'})
fig,axs=plt.subplots(1,3,figsize=(16,7))
results=[]
for ax,s,label in zip(axs,sigmas,['Wide biological range','Intermediate biological range','Narrow biological range']):
 t=s*z;x=t+tech*er;y=t+tech*ep
 r=np.corrcoef(x,y)[0,1];R=s*s/(s*s+tech*tech)
 results.append((s,R,r))
 ax.scatter(x,y,s=20,color='#177A78',alpha=.6,edgecolor='none')
 lim=3.6*np.sqrt(s*s+tech*tech)
 ax.plot([-lim,lim],[-lim,lim],color='#a9b1b7',ls='--',lw=1.4,label='True relation')
 ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Measured RNA (centered log2)',ylabel='Measured protein (centered log2)')
 ax.set_aspect('equal');ax.set_title(f'{label}\nBiological SD = {s:.2f}',fontsize=16,pad=15)
 ax.text(.04,.96,f'Observed sample r = {r:.2f}\nPopulation r = {R:.3f}',transform=ax.transAxes,va='top',fontsize=14,bbox={'facecolor':'white','edgecolor':'none','alpha':.88})
fig.suptitle('Perfect biological coupling can yield almost no observed correlation',fontsize=24,y=.96)
fig.text(.5,.865,'True RNA = true protein in every cell. Independent measurement-error SD = 0.20 in both modalities.',ha='center',fontsize=15)
fig.text(.5,.08,'Illustrative simulation, not paper data. 318 cells per panel. Axis limits differ; units and error SD are identical.',ha='center',fontsize=13)
fig.subplots_adjust(left=.065,right=.98,bottom=.19,top=.76,wspace=.32)
for ext in ['png','svg']:fig.savefig(OUT/f'01_perfect_coupling.{ext}',dpi=200)
plt.close(fig)
# Sampling distribution: 2,000 independent experiments per biological SD.
sgrid=np.geomspace(.01,1,50);reps=2000;avg=[];lo=[];hi=[]
for s in sgrid:
 t=rng.normal(0,s,(reps,n));x=t+rng.normal(0,tech,(reps,n));y=t+rng.normal(0,tech,(reps,n))
 x-=x.mean(axis=1,keepdims=True);y-=y.mean(axis=1,keepdims=True)
 rs=(x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))
 avg.append(rs.mean());l,h=np.quantile(rs,[.025,.975]);lo.append(l);hi.append(h)
expected=sgrid**2/(sgrid**2+tech**2)
assert np.max(np.abs(np.array(avg)-expected))<.01
fig,ax=plt.subplots(figsize=(13,7))
ax.fill_between(sgrid,lo,hi,color='#177A78',alpha=.18,label='Central 95% of simulated sample correlations')
ax.plot(sgrid,expected,color='#177A78',lw=3,label='Population correlation predicted by reliability')
ax.scatter(sgrid,avg,s=14,color='#152936',label='Mean across 2,000 simulated experiments')
ax.axhline(0,color='#89969e',lw=1);ax.axhline(1,color='#89969e',lw=1,ls='--')
ax.set(xscale='log',ylim=(-.17,1.06),xlabel='True biological SD (log2 abundance)',ylabel='Observed Pearson correlation')
ax.set_title('Biological range determines the attainable correlation',fontsize=22,pad=22)
ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='white',framealpha=1,fontsize=12)
fig.text(.5,.025,'True correlation = 1. Error SD = 0.20 in both modalities. n = 318. Shading is a sampling interval, not a fitted confidence band.',ha='center',fontsize=11)
fig.tight_layout(rect=[.02,.06,.98,.98])
for ext in ['png','svg']:fig.savefig(OUT/f'02_noise_ceiling.{ext}',dpi=200)
plt.close(fig)
np.savetxt(OUT/'simulation_summary.csv',results,delimiter=',',header='biological_log2_SD,reliability_and_population_r,sample_r',comments='')
print(results)
