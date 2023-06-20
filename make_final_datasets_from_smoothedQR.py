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
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import awkward as ak
import case_paths.jet_sample as js
import case_paths.util.sample_factory as sf
import case_paths.util.experiment as ex
import case_paths.path_constants.sample_dict_file_parts_reco as sdfr
import case_paths.path_constants.sample_dict_file_parts_selected as sdfs
import case_qr_fixed.selection.discriminator as disc
import case_qr_fixed.selection.loss_strategy as lost
import case_qr_fixed.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import case_qr_fixed.util.data_processing as dapr
import case_paths.phase_space.cut_constants as cuts
import pathlib
sns.set_context("paper")
import pdb
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

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

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
case_qr_results = '/work/abal/CASE/QR_results'
case_qr_models = '/work/abal/CASE/QR_models'
#####
quan=15
# signal_name = sys.argv[1]
# #signal_injections = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# #signal_injections = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# signal_injections = [float(sys.argv[2])]
# outfolder = sys.argv[3]
# folds = int(sys.argv[4])
############

signal_name = 'WpToBpT_Wp3000_Bp400_Top170_ZbtReco'
qcd_reco_id = 'qcdSigMCOrigReco'
signal_injections = [0.0]
folds = 4
qcd_sample = 'qcdSigMCOrig'
run_n = 50005
##### Define directory paths
case_qr_results = '/work/abal/CASE/QR_results'
case_qr_models = '/work/abal/CASE/QR_models'

#####
quantiles = []
dec_quantiles=[]
for i in range(2,31,2):
    q_inv=100-i
    quantiles.append(str(q_inv))
    dec_quantiles.append(i/100)
quantiles = quantiles+['10', '05','01']
dec_quantiles=dec_quantiles+[0.9,0.95,0.99]

bin_low = 1455 
bin_high = 6455
srange=[bin_low,bin_high]
bins = np.linspace(bin_low,bin_high,(bin_high-bin_low)//2)
bin_centers = [(high+low)/2 for low, high in zip(bins[:-1], bins[1:])]

models={}
discriminator=disc.VQRv1Discriminator_KerasAPI(quantiles=dec_quantiles, loss_strategy=lost.loss_strategy_dict['rk5_05'], batch_sz=256, epochs=800,  n_layers=5, n_nodes=8)
disc_storage=f'/work/abal/CASE/QR_models/run_{run_n}/trained_models/{qcd_sample}'
        
out_dir=f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}/QR_dump/{qcd_reco_id}'

for siginj in signal_injections:
    print(f'Analysing data with {siginj} pb-1 of signal {signal_name}$')
    
    
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
        k_preds=[]
        
        # Load the background file for the k-th fold
        with h5py.File(os.path.join(out_dir,f'{qcd_reco_id}_fold_{k}.h5'), "r") as bkg_f:
                branch_names = bkg_f['eventFeatureNames'][()].astype(str)
                #print(branch_names)
                features = bkg_f['eventFeatures'][()]
                mask = features[:,0]<9000
                features = features[mask]
                mjj = np.asarray(features[:,0])
                loss_indices = get_loss_indices(branch_names)
                loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 
            
        # Load the trained Deep QRs for the folds other than k and apply them to the k-th fold
        for l in range(folds):
            if k==l: continue
            model_name = f'{qcd_sample}_fold_{l}_8nodes_{quan}quantiles.h5'
            discriminator.load(os.path.join(disc_storage,model_name))
    
            yss=np.array(discriminator.predict(mjj,flatten=False))
            k_preds.append(yss)
        k_preds=np.mean(np.stack([k_pred for k_pred in k_preds],axis=0),axis=0)
        
            
        #pdb.set_trace()    
        for j,q in enumerate(quantiles):
            qKey = f'all_sel_q'+q
            if qKey in all_sel_q:
                all_sel_q[qKey]=np.concatenate((all_sel_q[qKey],np.expand_dims((loss > k_preds[:,j]),axis=-1)))
            else:
                all_sel_q[qKey]= np.expand_dims((loss > k_preds[:,j]),axis=-1)
        
        if len(all_loss) == 0:
            all_loss = np.expand_dims(loss,axis=-1)
            all_mjj = np.expand_dims(mjj,axis=-1)
        else:
            all_loss = np.concatenate((all_loss,np.expand_dims(loss,axis=-1)))
            all_mjj = np.concatenate((all_mjj,np.expand_dims(mjj,axis=-1)))
    
        
        
    #all_event_features = np.concatenate((all_mjj,all_loss,all_sel_q70,all_sel_q50,all_sel_q30,all_sel_q10,all_sel_q05,all_sel_q01),axis=-1)
    dt = h5py.special_dtype(vlen=str)
    outfilename = f'bkg_smoothing_{srange[0]}_{srange[1]}_{qcd_sample}'
    if siginj != 0:
        outfilename = 'bkg_%s_%s'%(qcd_reco_id,str(siginj))
    out_dir=f'/work/abal/CASE/QR_datasets/run_{run_n}/validation_tests/QRext/{quan}_quantiles'
    pathlib.Path(out_dir).mkdir(parents=True,exist_ok=True)
    hf = h5py.File(os.path.join(out_dir,f'{outfilename}.h5'), 'w')
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
    hf.close()

