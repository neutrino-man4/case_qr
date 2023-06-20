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
case_qr_results = '/storage/9/abal/CASE/QR_results'
case_qr_models = '/work/abal/CASE/QR_models'
#####

# signal_name = sys.argv[1]
# #signal_injections = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# #signal_injections = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# signal_injections = [float(sys.argv[2])]
# outfolder = sys.argv[3]
# folds = int(sys.argv[4])
############

signal_name = 'grav_3p5_naReco'
signal_injections = [0.01,0.1]
folds = 4
qcd_reco_id = 'delphes_bkg'
run_n = 50005
quan=30
methods=['ER','expectiles']#['QR','quantiles']# 
quantiles = []
for i in range(1,quan+1):
    q_inv=100-i
    if q_inv==58: continue
    quantiles.append(str(q_inv))
quantiles = quantiles+['10', '05','01']
srange = [1455,6500]


##### Define directory paths
case_storage = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/'
case_poly = f'/work/abal/CASE/QR_models/run_{run_n}/delphes_models_lmfit_csv/{qcd_reco_id}/smoothing_{srange[0]}_{srange[1]}_{methods[0]}/{quan}_{methods[1]}'
#####

for siginj in signal_injections:
    print(f'Analysing data with {siginj} pb-1 of signal {signal_name}$')
    
    case_poly_inj = f'/work/abal/CASE/QR_models/run_{run_n}/delphes_models_lmfit_csv/{signal_name}_{siginj}/smoothing_{srange[0]}_{srange[1]}_{methods[0]}/{quan}_{methods[1]}'
    
    all_mjj = []
    all_loss = []
    all_sel_q = {} # for all quantiles
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
    sig_sel_q = {}
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
                tmpdf = pd.read_csv(os.path.join(case_poly,f'bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv'))
                sig_tmpdf = pd.read_csv(os.path.join(case_poly,f'sig_lmfit_modelresult_quant_q{q}.csv'))
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            else:
                print('Loading QR trained on samples with injected signal')
                tmpdf = pd.read_csv(os.path.join(case_poly_inj,f'data_lmfit_modelresult_fold_{k}_quant_q{q}.csv'))
                sig_tmpdf = pd.read_csv(os.path.join(case_poly_inj,f'sig_lmfit_modelresult_quant_q{q}.csv'))
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
            #p = np.poly1d(tmpdf['par'].values.tolist())

            #print("Quantile", q)
            #print(tmpdf['par'].values.tolist()[::-1])
            #print(p(1200))
            qrs.append(p)
            
        with h5py.File(os.path.join(case_storage,f"{qcd_reco_id}_fold_{k}.h5"), "r") as bkg_f: # Irrespective of what its been trained on, the QR/ER is applied to the background-only dataset.
            branch_names = bkg_f['eventFeatureNames'][()].astype(str)
            #print(branch_names)
            features = bkg_f['eventFeatures'][()]
            mask = features[:,0]<9000
            features = features[mask]
            mjj = np.asarray(features[:,0])
            loss_indices = get_loss_indices(branch_names)
            loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

            for j,q in enumerate(quantiles):
                qKey = f'all_sel_q'+q
                if qKey in all_sel_q:
                    all_sel_q[qKey]=np.concatenate((all_sel_q[qKey],np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                else:
                    all_sel_q[qKey]= np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                # if q == '70':
                #     if len(all_sel_q70) == 0:
                #         all_sel_q70 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q70 = np.concatenate((all_sel_q70,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                # if q == '50':
                #     if len(all_sel_q50) == 0:
                #         all_sel_q50 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q50 = np.concatenate((all_sel_q50,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                # if q == '30':
                #     if len(all_sel_q30) == 0:
                #         all_sel_q30 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q30 = np.concatenate((all_sel_q30,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                # if q == '10':
                #     if len(all_sel_q10) == 0:
                #         all_sel_q10 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q10 = np.concatenate((all_sel_q10,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                # if q == '05':
                #     if len(all_sel_q05) == 0:
                #         all_sel_q05 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q05 = np.concatenate((all_sel_q05,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                # if q == '01':
                #     if len(all_sel_q01) == 0:
                #         all_sel_q01 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                #     else:
                #         all_sel_q01 = np.concatenate((all_sel_q01,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
            if len(all_loss) == 0:
                all_loss = np.expand_dims(loss,axis=-1)
                all_mjj = np.expand_dims(mjj,axis=-1)
            else:
                all_loss = np.concatenate((all_loss,np.expand_dims(loss,axis=-1)))
                all_mjj = np.concatenate((all_mjj,np.expand_dims(mjj,axis=-1)))
        
        
        # if k == 0:
        #         with h5py.File(f"/work/abal/CASE/QR_results/events/run_{run_n}/sig_{bkg_name}/xsec_0/loss_rk5_05/{signal_name}.h5", "r") as sig_f:
        #             branch_names = sig_f['eventFeatureNames'][()].astype(str)
        #             #print(branch_names)
        #             features = sig_f['eventFeatures'][()]
        #             mask = features[:,0]<9000
        #             features = features[mask]
        #             mjj = np.asarray(features[:,0])
                    
        #             loss_indices = get_loss_indices(branch_names)
        #             loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

        #             for j,q in enumerate(quantiles):
        #                 qKey = f'sig_sel_q'+q
        #                 if qKey in sig_sel_q:
        #                     sig_sel_q[qKey]=np.concatenate((sig_sel_q[qKey],np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 else:
        #                     sig_sel_q[qKey]= np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                
        #                 # if q == '30':
        #                 #     if len(sig_sel_q30) == 0:
        #                 #         sig_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q30 = np.concatenate((sig_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 # if q == '70':
        #                 #     if len(sig_sel_q70) == 0:
        #                 #         sig_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q70 = np.concatenate((sig_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 # if q == '50':
        #                 #     if len(sig_sel_q50) == 0:
        #                 #         sig_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q50 = np.concatenate((sig_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 # if q == '10':
        #                 #     if len(sig_sel_q10) == 0:
        #                 #         sig_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q10 = np.concatenate((sig_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 # if q == '05':
        #                 #     if len(sig_sel_q05) == 0:
        #                 #         sig_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q05 = np.concatenate((sig_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #                 # if q == '01':
        #                 #     if len(sig_sel_q01) == 0:
        #                 #         sig_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
        #                 #     else:
        #                 #         sig_sel_q01 = np.concatenate((sig_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        #             if len(sig_loss) == 0:
        #                 sig_loss = np.expand_dims(loss,axis=-1)
        #                 sig_mjj = np.expand_dims(mjj,axis=-1)
        #             else:
        #                 sig_loss = np.concatenate((sig_loss,np.expand_dims(loss,axis=-1)))
        #                 sig_mjj = np.concatenate((sig_mjj,np.expand_dims(mjj,axis=-1)))

    #all_event_features = np.concatenate((all_mjj,all_loss,all_sel_q70,all_sel_q50,all_sel_q30,all_sel_q10,all_sel_q05,all_sel_q01),axis=-1)
    dt = h5py.special_dtype(vlen=str)
    
    outdir=f'/ceph/abal/CASE/QR_datasets/run_{run_n}/delphes_tests/{methods[0]}/{quan}_{methods[1]}'
    import pathlib;pathlib.Path(outdir).mkdir(parents=True,exist_ok=True) # Create directory with parents if it does not exist. 
    
    outfilename = f'{qcd_reco_id}_smoothing_{srange[0]}_{srange[1]}'
    # file contains only bkg events and QR has also been trained on bkg only events. 
    if siginj != 0:
        outfilename = f'{qcd_reco_id}_smoothing_{srange[0]}_{srange[1]}_inj_{siginj}_{signal_name}'
        # file contains only bkg events but the QR applied to it has been trained on bkg+inj signal events.  
    hf = h5py.File(os.path.join(outdir,f'{outfilename}.h5'), 'w')
    hf.create_dataset('mjj', data=np.array(all_mjj))
    hf.create_dataset('loss', data=np.array(all_loss))
    for q in quantiles:
        q_name='sel_q'+str(100-int(q))
        hf.create_dataset(q_name, data=np.array(all_sel_q['all_sel_q'+q]))
        
    # hf.create_dataset('sel_q30', data=np.array(all_sel_q70))
    # hf.create_dataset('sel_q50', data=np.array(all_sel_q50))
    # hf.create_dataset('sel_q70', data=np.array(all_sel_q30))
    # hf.create_dataset('sel_q90', data=np.array(all_sel_q10))
    # hf.create_dataset('sel_q95', data=np.array(all_sel_q05))
    # hf.create_dataset('sel_q99', data=np.array(all_sel_q01))
    # hf.create_dataset('eventFeatures', data=np.array(all_event_features))
    # hf.create_dataset('eventFeatureNames', data=np.array(all_event_feature_names_reversed,dtype=dt))
    #import pdb; pdb.set_trace()
    hf.close()
    print(f'background file saved to ',os.path.join(outdir,f'{outfilename}.h5'))

    # if siginj == 0:
    #     #sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    #     sig_outfilename = 'signal_%s'%(signal_name)
    #     sig_hf = h5py.File(f'/work/abal/CASE/QR_datasets/run_{run_n}/validation_tests/QRext/{sig_outfilename}.h5', 'w')
    #     sig_hf.create_dataset('mjj', data=np.array(sig_mjj))
    #     sig_hf.create_dataset('loss', data=np.array(sig_loss))
    #     for q in quantiles:
    #         q_name='sel_q'+str(100-int(q))
    #         sig_hf.create_dataset(q_name, data=np.array(sig_sel_q['sig_sel_q'+q]))
    #     # sig_hf.create_dataset('sel_q30', data=np.array(sig_sel_q70))
    #     # sig_hf.create_dataset('sel_q50', data=np.array(sig_sel_q50))
    #     # sig_hf.create_dataset('sel_q70', data=np.array(sig_sel_q30))
    #     # sig_hf.create_dataset('sel_q90', data=np.array(sig_sel_q10))
    #     # sig_hf.create_dataset('sel_q95', data=np.array(sig_sel_q05))
    #     # sig_hf.create_dataset('sel_q99', data=np.array(sig_sel_q01))
    #     # sig_hf.create_dataset('eventFeatures', data=np.array(sig_event_features))
    #     # sig_hf.create_dataset('eventFeatureNames', data=np.array(sig_event_feature_names_reversed,dtype=dt))
    #     sig_hf.close()
    #     print('Signal file with quantile bool array has been saved')


        


