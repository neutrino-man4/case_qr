#!/usr/bin/env python
# coding: utf-8

# In[7]:

import tqdm
import sys
import os
#sys.path.insert(0, os.path.abspath('/eos/user/b/bmaier/.local/lib/python3.9/site-packages/'))
import h5py
import numpy as np
import awkward as ak
import uproot as uproot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import awkward as ak
sns.set_context("paper")
#print(ak.__version__)
#import mplhep as hep
import json
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
case_qr_results = '/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results'
case_qr_models = '/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models'
#####

# signal_name = sys.argv[1]
# #signal_injections = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# #signal_injections = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# signal_injections = [float(sys.argv[2])]
# outfolder = sys.argv[3]
# folds = int(sys.argv[4])
############

signal_name = 'WkkToWRadionToWWW_M3000_Mr170Reco'
signal_injections = np.arange(0,0.11,0.01)
folds = 4
qcd_sample = 'MCOrig_QR_Reco'
run_n = 28332
##### Define directory paths
case_qr_results = '/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results'
case_qr_models = '/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models'

#####

for siginj in signal_injections:
    print(f'Analysing data with {siginj} pb-1 of signal {signal_name}$')
    
    quantiles = ['70', '50', '30', '10', '05','01']
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

    for k in tqdm.tqdm(range(folds)):
        #print(f"Fold {k}")
        qrs = []
        sig_qrs = []
        for q in quantiles:
            if siginj == 0.:
                tmpdf = pd.read_csv(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models/run_{run_n}/models_lmfit_csv/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models/run_{run_n}/models_lmfit_csv/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            else:
                tmpdf = pd.read_csv(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models/run_{run_n}/models_lmfit_csv_{signal_name}_{str(siginj)}/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_models/run_{run_n}/models_lmfit_csv_{signal_name}_{str(siginj)}/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
            #p = np.poly1d(tmpdf['par'].values.tolist())

            #print("Quantile", q)
            #print(tmpdf['par'].values.tolist()[::-1])
            #print(p(1200))
            qrs.append(p)
            
        with h5py.File(f"/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/sig_{qcd_sample}_fold_{k}.h5", "r") as bkg_f:
            branch_names = bkg_f['eventFeatureNames'][()].astype(str)
            #print(branch_names)
            features = bkg_f['eventFeatures'][()]
            mask = features[:,0]<9000
            features = features[mask]
            mjj = np.asarray(features[:,0])
            loss_indices = get_loss_indices(branch_names)
            loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

            for j,q in enumerate(quantiles):
                if q == '70':
                    if len(all_sel_q70) == 0:
                        all_sel_q70 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q70 = np.concatenate((all_sel_q70,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '50':
                    if len(all_sel_q50) == 0:
                        all_sel_q50 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q50 = np.concatenate((all_sel_q50,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '30':
                    if len(all_sel_q30) == 0:
                        all_sel_q30 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q30 = np.concatenate((all_sel_q30,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '10':
                    if len(all_sel_q10) == 0:
                        all_sel_q10 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q10 = np.concatenate((all_sel_q10,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '05':
                    if len(all_sel_q05) == 0:
                        all_sel_q05 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q05 = np.concatenate((all_sel_q05,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '01':
                    if len(all_sel_q01) == 0:
                        all_sel_q01 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q01 = np.concatenate((all_sel_q01,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
            if len(all_loss) == 0:
                all_loss = np.expand_dims(loss,axis=-1)
                all_mjj = np.expand_dims(mjj,axis=-1)
            else:
                all_loss = np.concatenate((all_loss,np.expand_dims(loss,axis=-1)))
                all_mjj = np.concatenate((all_mjj,np.expand_dims(mjj,axis=-1)))

        if siginj == 0.:
            if k == 0:
                with h5py.File(f"/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{signal_name}.h5", "r") as sig_f:
                    branch_names = sig_f['eventFeatureNames'][()].astype(str)
                    #print(branch_names)
                    features = sig_f['eventFeatures'][()]
                    mask = features[:,0]<9000
                    features = features[mask]
                    mjj = np.asarray(features[:,0])
                    loss_indices = get_loss_indices(branch_names)
            
                    loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

                    for j,q in enumerate(quantiles):
                        if q == '30':
                            if len(sig_sel_q30) == 0:
                                sig_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q30 = np.concatenate((sig_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '70':
                            if len(sig_sel_q70) == 0:
                                sig_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q70 = np.concatenate((sig_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '50':
                            if len(sig_sel_q50) == 0:
                                sig_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q50 = np.concatenate((sig_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '10':
                            if len(sig_sel_q10) == 0:
                                sig_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q10 = np.concatenate((sig_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '05':
                            if len(sig_sel_q05) == 0:
                                sig_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q05 = np.concatenate((sig_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '01':
                            if len(sig_sel_q01) == 0:
                                sig_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q01 = np.concatenate((sig_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                    if len(sig_loss) == 0:
                        sig_loss = np.expand_dims(loss,axis=-1)
                        sig_mjj = np.expand_dims(mjj,axis=-1)
                    else:
                        sig_loss = np.concatenate((sig_loss,np.expand_dims(loss,axis=-1)))
                        sig_mjj = np.concatenate((sig_mjj,np.expand_dims(mjj,axis=-1)))

        if siginj != 0.:
            with h5py.File(f"/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/data_{signal_name}_{siginj}_fold_{k}.h5", "r") as data_f:
                branch_names = data_f['eventFeatureNames'][()].astype(str)
                #print(branch_names)
                features = data_f['eventFeatures'][()]
                mask = features[:,0]<9000
                features = features[mask]
                mjj = np.asarray(features[:,0])
                loss_indices = get_loss_indices(branch_names)
            
                loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

                for j,q in enumerate(quantiles):
                    if q == '30':
                        if len(data_sel_q30) == 0:
                            data_sel_q30 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q30 = np.concatenate((data_sel_q30,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '70':
                        if len(data_sel_q70) == 0:
                            data_sel_q70 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q70 = np.concatenate((data_sel_q70,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '50':
                        if len(data_sel_q50) == 0:
                            data_sel_q50 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q50 = np.concatenate((data_sel_q50,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '10':
                        if len(data_sel_q10) == 0:
                            data_sel_q10 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q10 = np.concatenate((data_sel_q10,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '05':
                        if len(data_sel_q05) == 0:
                            data_sel_q05 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q05 = np.concatenate((data_sel_q05,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '01':
                        if len(data_sel_q01) == 0:
                            data_sel_q01 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q01 = np.concatenate((data_sel_q01,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if len(data_loss) == 0:
                    data_loss = np.expand_dims(loss,axis=-1)
                    data_mjj = np.expand_dims(mjj,axis=-1)
                else:
                    data_loss = np.concatenate((data_loss,np.expand_dims(loss,axis=-1)))
                    data_mjj = np.concatenate((data_mjj,np.expand_dims(mjj,axis=-1)))

            if k == 0:
                with h5py.File(f"/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{signal_name}.h5", "r") as sig_f:
                    branch_names = sig_f['eventFeatureNames'][()].astype(str)
                    #print(branch_names)
                    features = sig_f['eventFeatures'][()]
                    mask = features[:,0]<9000
                    features = features[mask]
                    mjj = np.asarray(features[:,0])
                    
                    loss_indices = get_loss_indices(branch_names)
                    loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

                    for j,q in enumerate(quantiles):
                        if q == '30':
                            if len(sig_sel_q30) == 0:
                                sig_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q30 = np.concatenate((sig_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '70':
                            if len(sig_sel_q70) == 0:
                                sig_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q70 = np.concatenate((sig_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '50':
                            if len(sig_sel_q50) == 0:
                                sig_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q50 = np.concatenate((sig_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '10':
                            if len(sig_sel_q10) == 0:
                                sig_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q10 = np.concatenate((sig_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '05':
                            if len(sig_sel_q05) == 0:
                                sig_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q05 = np.concatenate((sig_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '01':
                            if len(sig_sel_q01) == 0:
                                sig_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                sig_sel_q01 = np.concatenate((sig_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                    if len(sig_loss) == 0:
                        sig_loss = np.expand_dims(loss,axis=-1)
                        sig_mjj = np.expand_dims(mjj,axis=-1)
                    else:
                        sig_loss = np.concatenate((sig_loss,np.expand_dims(loss,axis=-1)))
                        sig_mjj = np.concatenate((sig_mjj,np.expand_dims(mjj,axis=-1)))


                with h5py.File(f"/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/injected_{signal_name}_{siginj}.h5", "r") as inj_f:
                    branch_names = inj_f['eventFeatureNames'][()].astype(str)
                    #print(branch_names)
                    features = inj_f['eventFeatures'][()]
                    mask = features[:,0]<9000
                    features = features[mask]
                    mjj = np.asarray(features[:,0])
                    loss_indices = get_loss_indices(branch_names)
            
                    loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

                    for j,q in enumerate(quantiles):
                        if q == '30':
                            if len(inj_sel_q30) == 0:
                                inj_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q30 = np.concatenate((inj_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '70':
                            if len(inj_sel_q70) == 0:
                                inj_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q70 = np.concatenate((inj_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '50':
                            if len(inj_sel_q50) == 0:
                                inj_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q50 = np.concatenate((inj_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '10':
                            if len(inj_sel_q10) == 0:
                                inj_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q10 = np.concatenate((inj_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '05':
                            if len(inj_sel_q05) == 0:
                                inj_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q05 = np.concatenate((inj_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                        if q == '01':
                            if len(inj_sel_q01) == 0:
                                inj_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                            else:
                                inj_sel_q01 = np.concatenate((inj_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
                    if len(inj_loss) == 0:
                        inj_loss = np.expand_dims(loss,axis=-1)
                        inj_mjj = np.expand_dims(mjj,axis=-1)
                    else:
                        inj_loss = np.concatenate((inj_loss,np.expand_dims(loss,axis=-1)))
                        inj_mjj = np.concatenate((inj_mjj,np.expand_dims(mjj,axis=-1)))


    all_event_features = np.concatenate((all_mjj,all_loss,all_sel_q70,all_sel_q50,all_sel_q30,all_sel_q10,all_sel_q05,all_sel_q01),axis=-1)
    if siginj == 0:
        sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    print(all_event_features.shape)
    if siginj != 0:
        data_event_features = np.concatenate((data_mjj,data_loss,data_sel_q70,data_sel_q50,data_sel_q30,data_sel_q10,data_sel_q05,data_sel_q01),axis=-1)
        print(data_event_features.shape)
        sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
        print(sig_event_features.shape)
        inj_event_features = np.concatenate((inj_mjj,inj_loss,inj_sel_q70,inj_sel_q50,inj_sel_q30,inj_sel_q10,inj_sel_q05,inj_sel_q01),axis=-1)
    
    outfilename = 'bkg'
    if siginj != 0:
        outfilename = 'bkg_%s_%s'%(signal_name,str(siginj))
    hf = h5py.File(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_datasets/run_{run_n}/{outfilename}.h5', 'w')
    hf.create_dataset('mjj', data=np.array(all_mjj))
    hf.create_dataset('loss', data=np.array(all_loss))
    hf.create_dataset('sel_q30', data=np.array(all_sel_q70))
    hf.create_dataset('sel_q50', data=np.array(all_sel_q50))
    hf.create_dataset('sel_q70', data=np.array(all_sel_q30))
    hf.create_dataset('sel_q90', data=np.array(all_sel_q10))
    hf.create_dataset('sel_q95', data=np.array(all_sel_q05))
    hf.create_dataset('sel_q99', data=np.array(all_sel_q01))
    hf.create_dataset('eventFeatures', data=np.array(all_event_features))
    hf.create_dataset('eventFeatureNames', data=all_event_feature_names_reversed)
    hf.close()
    if siginj == 0:
        sig_outfilename = 'signal_%s'%(signal_name)
        sig_hf = h5py.File(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_datasets/run_{run_n}/{sig_outfilename}.h5', 'w')
        sig_hf.create_dataset('mjj', data=np.array(sig_mjj))
        sig_hf.create_dataset('loss', data=np.array(sig_loss))
        sig_hf.create_dataset('sel_q30', data=np.array(sig_sel_q70))
        sig_hf.create_dataset('sel_q50', data=np.array(sig_sel_q50))
        sig_hf.create_dataset('sel_q70', data=np.array(sig_sel_q30))
        sig_hf.create_dataset('sel_q90', data=np.array(sig_sel_q10))
        sig_hf.create_dataset('sel_q95', data=np.array(sig_sel_q05))
        sig_hf.create_dataset('sel_q99', data=np.array(sig_sel_q01))
        sig_hf.create_dataset('eventFeatures', data=np.array(sig_event_features))
        sig_hf.create_dataset('eventFeatureNames', data=sig_event_feature_names_reversed)
        sig_hf.close()
        
    if siginj != 0:
        data_outfilename = 'data_%s_%s'%(signal_name,str(siginj))
        data_hf = h5py.File(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_datasets/run_{run_n}/{data_outfilename}.h5', 'w')
        data_hf.create_dataset('mjj', data=np.array(data_mjj))
        data_hf.create_dataset('loss', data=np.array(data_loss))
        data_hf.create_dataset('sel_q30', data=np.array(data_sel_q70))
        data_hf.create_dataset('sel_q50', data=np.array(data_sel_q50))
        data_hf.create_dataset('sel_q70', data=np.array(data_sel_q30))
        data_hf.create_dataset('sel_q90', data=np.array(data_sel_q10))
        data_hf.create_dataset('sel_q95', data=np.array(data_sel_q05))
        data_hf.create_dataset('sel_q99', data=np.array(data_sel_q01))
        data_hf.create_dataset('eventFeatures', data=np.array(data_event_features))
        data_hf.create_dataset('eventFeatureNames', data=data_event_feature_names_reversed)
        data_hf.close()
        plt.plot()

        sig_outfilename = 'signal_%s_%s'%(signal_name,str(siginj))
        sig_hf = h5py.File(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_datasets/run_{run_n}/{sig_outfilename}.h5', 'w')
        sig_hf.create_dataset('mjj', data=np.array(sig_mjj))
        sig_hf.create_dataset('loss', data=np.array(sig_loss))
        sig_hf.create_dataset('sel_q30', data=np.array(sig_sel_q70)) # Most QCD like
        sig_hf.create_dataset('sel_q50', data=np.array(sig_sel_q50))
        sig_hf.create_dataset('sel_q70', data=np.array(sig_sel_q30))
        sig_hf.create_dataset('sel_q90', data=np.array(sig_sel_q10))
        sig_hf.create_dataset('sel_q95', data=np.array(sig_sel_q05))
        sig_hf.create_dataset('sel_q99', data=np.array(sig_sel_q01)) # Most BSM like
        sig_hf.create_dataset('eventFeatures', data=np.array(sig_event_features))
        sig_hf.create_dataset('eventFeatureNames', data=sig_event_feature_names_reversed)
        sig_hf.close()


        inj_outfilename = 'injected_%s_%s'%(signal_name,str(siginj))
        inj_hf = h5py.File(f'/eos/uscms/store/user/izoi/CASE/CASE_Feb2023/QR_datasets/run_{run_n}/{inj_outfilename}.h5', 'w')
        inj_hf.create_dataset('mjj', data=np.array(inj_mjj))
        inj_hf.create_dataset('loss', data=np.array(inj_loss))
        inj_hf.create_dataset('sel_q30', data=np.array(inj_sel_q70))
        inj_hf.create_dataset('sel_q50', data=np.array(inj_sel_q50))
        inj_hf.create_dataset('sel_q70', data=np.array(inj_sel_q30))
        inj_hf.create_dataset('sel_q90', data=np.array(inj_sel_q10))
        inj_hf.create_dataset('sel_q95', data=np.array(inj_sel_q05))
        inj_hf.create_dataset('sel_q99', data=np.array(inj_sel_q01))
        inj_hf.create_dataset('eventFeatures', data=np.array(inj_event_features))
        inj_hf.create_dataset('eventFeatureNames', data=inj_event_feature_names_reversed)
        inj_hf.close()

        


