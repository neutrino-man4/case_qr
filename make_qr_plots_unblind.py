#!/usr/bin/env python
# coding: utf-8

# In[7]:

import tqdm
import sys
import os
#sys.path.insert(0, os.path.abspath('/eos/user/b/bmaier/.local/lib/python3.9/site-packages/'))
import h5py
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#print(ak.__version__)
#import mplhep as hep
import json
matplotlib.use('TKAgg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
#plt.style.use(hep.style.CMS)

#import ROOT
#from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile
#from ROOT import gROOT, gBenchmark


# In[8]:


from lmfit.model import load_model


# In[9]:

def get_loss_indices(branch_names):
    j1_reco = np.argwhere(branch_names=='j1RecoLoss')[0,0]
    j1_kl = np.argwhere(branch_names=='j1KlLoss')[0,0]
    j2_reco = np.argwhere(branch_names=='j2RecoLoss')[0,0]
    j2_kl = np.argwhere(branch_names=='j2KlLoss')[0,0]
    loss_dict = {'j1KlLoss':j1_kl,'j1RecoLoss':j1_reco,'j2KlLoss':j2_kl,'j2RecoLoss':j2_reco}
    return loss_dict
    
def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))


# In[30]:
##### Define directory paths

# signal_name = sys.argv[1]
# #signal_injections = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# #signal_injections = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# signal_injections = [float(sys.argv[2])]
# outfolder = sys.argv[3]
# folds = int(sys.argv[4])
############

signal_name = 'WpToBpT_Wp3000_Bp400_Top170_ZbtReco'
signal_injections = [0.]
folds = 4
qcd_sample = 'qcdSRData'
run_n = 141098
##### Define directory paths
case_qr_results = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}'
case_qr_models = '/work/abal/CASE/QR_models'

#####

for siginj in signal_injections:
    print(f'Analysing data with {siginj} pb-1 of signal {signal_name}$')
    
    quantiles = ['70', '50', '30', '10', '05','01']
    inv_quantiles=['Q30','Q50','Q70','Q90','Q95','Q99']
    all_mjj = []
    all_loss = []
    all_sel_q70 = []
    all_sel_q50 = []
    all_sel_q30 = []
    all_sel_q10 = []
    all_sel_q05 = []
    all_sel_q01 = []
    all_event_features = []
    all_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    data_mjj = []
    data_loss = []
    data_sel_q70 = []
    data_sel_q50 = []
    data_sel_q30 = []
    data_sel_q10 = []
    data_sel_q05 = []
    data_sel_q01 = []
    data_event_features = []
    data_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    sig_mjj = []
    sig_loss = []
    sig_sel_q70 = []
    sig_sel_q50 = []
    sig_sel_q30 = []
    sig_sel_q10 = []
    sig_sel_q05 = []
    sig_sel_q01 = []
    sig_event_features = []
    sig_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    inj_mjj = []
    inj_loss = []
    inj_sel_q70 = []
    inj_sel_q50 = []
    inj_sel_q30 = []
    inj_sel_q10 = []
    inj_sel_q05 = []
    inj_sel_q01 = []
    inj_event_features = []
    inj_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']
    bin_low = 1455 
    bin_high = 5500
    bins = np.linspace(bin_low,bin_high,bin_high-bin_low)
    fig,ax=plt.subplots(2,2)
    ax=ax.flatten()
    fig.set_size_inches(18, 10)
    #ax[0].set(ylabel='Min. Loss $(j_{1},j_{2})$')
    #ax[2].set(ylabel='Min. Loss $(j_{1},j_{2})$')
    ax[0].set(ylabel='Loss ratio $Q/Q30$')
    ax[2].set(ylabel='Loss ratio $Q/Q30$')
        
    for k in tqdm.tqdm(range(folds)):
        if k>1:
            ax[k].set(xlabel='$m_{JJ}$')
        #print(f"Fold {k}")
        qrs = []
        sig_qrs = []
        for q in quantiles:
            if siginj == 0.:
                tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv/unblind_data_SR/1/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv/unblind_data_SR/1/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            else:
                print('This block should not be running at all. Exiting.')
                sys.exit(0)
                # Note that this section is redundant, should never be executed for unblinding since we don't do any injection tests here. 
                tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv_{signal_name}_{str(siginj)}/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv_{signal_name}_{str(siginj)}/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
            #p = np.poly1d(tmpdf['par'].values.tolist())

            #print("Quantile", q)
            #print(tmpdf['par'].values.tolist()[::-1])
            #print(p(1200))
            qrs.append(p)
        with h5py.File(f"{case_qr_results}/sig_{signal_name}/xsec_0/loss_rk5_05/1/sig_{qcd_sample}_fold_{k}.h5", "r") as bkg_f:
            branch_names = bkg_f['eventFeatureNames'][()].astype(str)
            #print(branch_names)
            features = bkg_f['eventFeatures'][()]
            mask = features[:,0]<9000
            features = features[mask]
            mjj = np.asarray(features[:,0])
            loss_indices = get_loss_indices(branch_names)
            loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 
            #ax[k].scatter(mjj,loss,label='Loss',s=1)
            
            for j,iq in enumerate(inv_quantiles):
                
                ax[k].plot(bins,qrs[j](bins)/qrs[0](bins),label=f'{iq}/Q30')
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',ncol=8)
    fig.savefig(f'plots/QR_unblindedData_qRatios.png',dpi=600)
    fig.clf()
    
    


