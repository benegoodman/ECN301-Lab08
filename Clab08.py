# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:07:29 2022

@author: bened
"""

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as norm
from scipy.stats import norm
import sys
import os

path = 'definepath'

os.chdir(path)

#
# read data from csv-file
df = pd.read_csv('hmda.csv')
print(df.info())

#%%

print(df['approv'].value_counts())

print(df['approv'].mean())

print(df['pi_rat'].describe())

#%%

#
# estimate probit model
probitmod = smf.probit(formula='approv ~ pi_rat', data=df)
probitres = probitmod.fit()
print(probitres.summary())

#%%

probitape = probitres.get_margeff()
print(probitape.summary())
papetab = pd.DataFrame(
    {'pro_b'   : probitape.margeff,
     'pro_se'  : probitape.margeff_se})

#%%

#
# format the confusion matrix
cm_df = pd.DataFrame(probitres.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1: 'Actual 1'})
print(cm_df)


# Model Accuracy
cm = np.array(cm_df)
ac = 100*(cm[0,0]+cm[1,1])/cm.sum()
print()
print('The model accuracy is {:.4}'.format(ac))
print()
print('Baseline accuracy is {:.4}'.format(100*df['approv'].mean()))
print()


#%%

#
# create plot of the CDF for the probit model

#
# 0 <= pi_rat <= 3
x = np.linspace(0,3)
y = probitres.params[0] + probitres.params[1]*x

fig,ax = plt.subplots(figsize=(4,4))
ax.plot(x,norm.cdf(y))
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()


#%%

#
# 0 <= pi_rat <= 2 (better graph)
x = np.linspace(0,2)
y = probitres.params[0] + probitres.params[1]*x
probcdf = norm.cdf(y)

fig,ax = plt.subplots(figsize=(5,4))
ax.plot(x,probcdf)
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()


#%%


#
# estimate logit model
logitmod = smf.logit(formula='approv ~ pi_rat', data=df)
logitres = logitmod.fit()
print(logitres.summary())

#
# marginal effects
logitape = logitres.get_margeff()
print(logitape.summary())
lapetab = pd.DataFrame(
    {'log_b'   : logitape.margeff,
     'log_se'  : logitape.margeff_se})

#
# format the confusion matrix
cm_df = pd.DataFrame(logitres.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1: 'Actual 1'})
print(cm_df)

# Model Accuracy
cm = np.array(cm_df)
ac = 100*(cm[0,0]+cm[1,1])/cm.sum()
print()
print('The model accuracy is {:.4}'.format(ac))
print('Baseline  accuracy is {:.4}'.format(100*df['approv'].mean()))
print()


#%%

#
# 0 <= pi_rat <= 2 (better graph)
x = np.linspace(0,2)
y = logitres.params[0] + logitres.params[1]*x
logcdf = norm.cdf(y)

fig,ax = plt.subplots(figsize=(5,4))
ax.plot(x,logcdf)
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()


#%%

#
# estimate linear probability model
linprmod = smf.ols(formula='approv ~ pi_rat', data=df)
linprres = linprmod.fit(cov_type='HC3')
print(linprres.summary())

#
# keep lpm estimate
linprtab = pd.DataFrame(
    {'lpm_b'   : round(linprres.params, 5),
     'lpm_se'  : round(linprres.bse, 5)})


#%%

#
# 0 <= pi_rat <= 2 (better graph)
x = np.linspace(0,2)
lincdf = linprres.params[0] + linprres.params[1]*x


fig,ax = plt.subplots(figsize=(5,4))
ax.plot(x,lincdf)
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()

#%%

#
# compare
fig,ax = plt.subplots(figsize=(5,4))
ax.plot(x,probcdf)
ax.plot(x,logcdf)
ax.plot(x,lincdf)
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.legend(['probit','logit','linear'])
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()

#%%

#
# compare APE results (Wooldridge Table 17.2)

#
# need some fancy pandas footwork here
#  (marginal effects are not indexed with variable names)
linprtab = linprtab.drop(['Intercept'])     # drop the intercept for the LPM
logprob = lapetab.join(papetab)             # merge logit and probit results
logprob = logprob.set_index(linprtab.index) # use the same index

apetab = pd.concat([linprtab, logprob], axis=1)
print(apetab)
print()

#%%

#
# estimate probit model
#
probit2mod = smf.probit(formula='approv ~ pi_rat + white', data=df)
probit2res = probit2mod.fit()
print(probit2res.summary())

#
# marginal effects
probit2ape = probit2res.get_margeff(dummy=True)
print(probit2ape.summary())
papetab2 = pd.DataFrame(
    {'pro_b'   : probit2ape.margeff,
     'pro_se'  : probit2ape.margeff_se})

#
# format the confusion matrix
cm_df = pd.DataFrame(probit2res.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1: 'Actual 1'})
print(cm_df)

# Model Accuracy
cm = np.array(cm_df)
ac = 100*(cm[0,0]+cm[1,1])/cm.sum()
print()
print('The model accuracy is {:.4}'.format(ac))
print('Baseline  accuracy is {:.4}'.format(100*df['approv'].mean()))
print()

#%%

# estimate probit model
#
pro_res_wo = smf.probit(formula='approv ~ pi_rat', data=df[df['white']==1]).fit()
print(pro_res_wo.summary())

pro_res_nw = smf.probit(formula='approv ~ pi_rat', data=df[df['white']==0]).fit()
print(pro_res_nw.summary())

#%%

#
# compare
wo = pro_res_wo.params[0] + pro_res_wo.params[1]*x
cdf_wo = norm.cdf(wo)
nw = pro_res_nw.params[0] + pro_res_nw.params[1]*x
cdf_nw = norm.cdf(nw)

fig,ax = plt.subplots(figsize=(5,4))
ax.plot(x,cdf_wo)
ax.plot(x,cdf_nw)
ax.set_ylabel('Prob of approval')
ax.set_xlabel('PI Ratio')
ax.legend(['White','Non-white'])
ax.grid()
fig.tight_layout()
#plt.savefig('fig1.png')
plt.show()

#%%

#
# estimate probit model
#
probit3mod = smf.probit(formula='approv ~ pi_rat + white + hse_inc + ltv_med + ltv_high + ccred + mcred + pubrec + denpmi + selfemp', data=df)
probit3res = probit3mod.fit()
print(probit3res.summary())

#
# marginal effects
probit3ape = probit3res.get_margeff(dummy=True)
print(probit3ape.summary())
papetab3 = pd.DataFrame(
    {'pro_b'   : probit3ape.margeff,
     'pro_se'  : probit3ape.margeff_se})

#
# format the confusion matrix
cm_df = pd.DataFrame(probit3res.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1: 'Actual 1'})
print(cm_df)

# Model Accuracy
cm = np.array(cm_df)
ac = 100*(cm[0,0]+cm[1,1])/cm.sum()
print()
print('The model accuracy is {:.4}'.format(ac))
print('Baseline  accuracy is {:.4}'.format(100*df['approv'].mean()))
print()

#%%

def do_gof(x_obs, x_pred):
    #
    # goodness of fit 
    #
    my = np.mean(x_obs)
    n_obs = len(x_obs)

    #
    # classification table
    c11 = np.sum(x_obs[x_pred>=0.5])
    c10 = np.sum(x_obs[x_pred< 0.5])
    c00 = np.sum((1 - x_obs)[x_pred< 0.5])
    c01 = np.sum((1 - x_obs)[x_pred>=0.5])

    #
    # baseline prediction
    bpi = 100.0*my
    if bpi < 50.0:
        bpi = 100.0 - bpi
    #
    # model accuracy
    macc = 100.0*(c00+c11)/n_obs
    #
    # model precision
    mpr0 = 100.0*c00/(c00+c10)
    mpr1 = 100.0*c11/(c01+c11)
    #
    # model recall
    mrc0 = 100.0*c00/(c00+c01)
    mrc1 = 100.0*c11/(c10+c11)

    #
    # this is cut and paste from somewhere
    # could use a filehandle f here (or edit to use print):    
    f = sys.stdout
    
    #
    f.write("==========================================\n")
    f.write("                Goodness of fit statistics\n")
    f.write("==========================================\n")
    f.write("Number of observations  : %16d\n" % (n_obs))
    f.write("Baseline predictions    : %16.2f\n" % (bpi))
    f.write("Model accuracy          : %16.2f\n" % (macc))
    f.write("Model precision (pos)   : %16.2f\n" % (mpr1))
    f.write("Model precision (neg)   : %16.2f\n" % (mpr0))
    f.write("Recall (sensitivity)    : %16.2f\n" % (mrc1))
    f.write("Recall (specificity)    : %16.2f\n" % (mrc0))
    f.write("==========================================\n\n")

    f.write("==========================================\n")
    f.write("   Classification table (Confusion Matrix)\n")
    f.write("==========================================\n")
    f.write("            |         Predicted |         \n")
    f.write("   Observed |        0        1 |    Total\n")
    f.write("------------+-------------------+---------\n")
    f.write("          0 | %8d %8d | %8d\n" % (c00, c01, c00+c01))
    f.write("          1 | %8d %8d | %8d\n" % (c10, c11, c10+c11))
    f.write("------------+-------------------+---------\n")
    f.write("      Total | %8d %8d | %8d\n" % (c00+c10, c01+c11, n_obs))
    f.write("------------+-------------------+---------\n\n")

            
#
do_gof(df['approv'],probit3res.predict())

#%%

#
# prepare data for the ROC graph
#
def get_roc(x_obs, x_prd):
    #
    # calculate the ROC graph
    #

    # Note: probably use numpy and pandas
    #       rewrite code at some time

    #
    # gather data
    #
    dfx = pd.DataFrame(columns=['obs'])
    dfx['obs']  = x_obs
    dfx['pred'] = x_prd
    dfx = dfx.sort_values('pred',ascending=False).copy()
    
    #
    # construct the ROC
    rlist = []
    pprev = 2
    tp = 0
    tn = 0
    for i, v in dfx.iterrows():
        if v['pred'] < pprev:
            rlist.append([tp,tn])
            pprev = v['pred']
        if v['obs'] > 0:
            tp += 1
        else:
            tn += 1
        rlist.append([tp,tn])

    #
    # convert to pandas dataframe
    roc = pd.DataFrame(rlist,columns=['tpr','tnr'])
    roc['tpr'] = roc['tpr']/roc['tpr'].iloc[-1]
    roc['tnr'] = roc['tnr']/roc['tnr'].iloc[-1]

    return roc

#
#roc = get_roc(df['approv'],probit3res.predict())

#%%

#
# ROC data and graph
proc = get_roc(df['approv'],probit3res.predict())

fig,ax = plt.subplots(figsize=(4,4))
ax.plot(proc.tnr,proc.tpr)
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
ax.grid()
fig.tight_layout()
#lt.savefig('probit_roc.png')
plt.show()

#%%

#
# function for the area under the ROC graph
#

def get_aur(x_obs, x_prd):
    #
    # calculate area under the ROC graph
    #

    # Note: probably use numpy and pandas
    #       rewrite code at some time

    #
    # gather data
    #
    dfx = pd.DataFrame(columns=['obs'])
    dfx['obs']  = x_obs
    dfx['pred'] = x_prd
    dfx = dfx.sort_values('pred',ascending=False).copy()
    
    #
    # construct the ROC
    pprev = 2
    tpc = 0
    tnc = 0
    tpp = 0
    tnp = 0
    aur = 0
    for i, v in dfx.iterrows():
        if v['pred'] < pprev:
            aur += abs(tnc-tnp)*(tpc+tpp)*0.5
            pprev = v['pred']
            tpp = tpc
            tnp = tnc
        if v['obs'] > 0:
            tpc += 1
        else:
            tnc += 1
    aur += abs(tnc-tnp)*(tpc+tpp)*0.5
    aur /= (tpc*tnc)

    return aur

get_aur(df['approv'],probit3res.predict())
