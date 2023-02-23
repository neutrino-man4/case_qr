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
import pathlib
from recordtype import recordtype

import case_paths.jet_sample as js
import case_paths.util.sample_factory as sf
import case_paths.util.experiment as ex
import case_paths.path_constants.sample_dict_file_parts_reco as sdfr
import case_paths.path_constants.sample_dict_file_parts_selected as sdfs
import case_qr.selection.discriminator as disc
import case_qr.selection.loss_strategy as lost
import case_qr.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import case_qr.util.data_processing as dapr
import case_paths.phase_space.cut_constants as cuts

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
case_qr_results = '/work/abal/CASE/QR_results'
case_qr_models = '/work/abal/CASE/QR_models'
#####

# signal_name = sys.argv[1]
# #signal_injections = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# #signal_injections = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# signal_injections = [float(sys.argv[2])]
# outfolder = sys.argv[3]
# folds = int(sys.argv[4])
############

signal_name = 'WkkToWRadionToWWW_M3000_Mr170Reco'
signal_folder = signal_name.replace('Reco','_RECO')

signal_injections = np.arange(0.1,0.11,0.01)
signal_injections=[0.1]
poly_order = 6
folds = 4
qcd_sample = 'MCOrig_QR_Reco'
run_n = 28332
##### Define directory paths
case_qr_results = '/work/abal/CASE/QR_results'
case_qr_models = '/work/abal/CASE/QR_models'
case_qr_data_mixed = f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}/qcd_sig_orig_RECO_mixed_jets'
case_qr_data_injected = f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}/qcd_sig_orig_RECO_injected_jets'

#####
Parameters = recordtype('Parameters','run_n, qcd_sample_id, sig_sample_id, strategy_id, epochs, kfold, poly_order, read_n')
params = Parameters(run_n=28332,
                    qcd_sample_id='qcdSigMCOrigReco',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    epochs=800,
                    kfold=4,
                    #kfold=3,
                    poly_order=6,
                    read_n=int(1e8))

dec_quantiles = [0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
discriminator=disc.VQRv1Discriminator_KerasAPI(quantiles=dec_quantiles, loss_strategy=lost.loss_strategy_dict[params.strategy_id]\
                                 , batch_sz=160, epochs=params.epochs,  n_layers=5, n_nodes=10)


for siginj in signal_injections:
    
    discriminator.load(f"{case_qr_models}/run_{run_n}/discriminator_WITH_MIXING_{signal_name}_inj_{siginj}.h5") #

    print(f'Analysing data with {siginj} pb of signal {signal_name}$')
    dt = h5py.special_dtype(vlen=str)

    quantiles = ['70', '50', '30', '10', '05','01'] # Actually the inverted quantiles
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

    data_event_feature_names_reversed = np.array(['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99'],dtype=dt)

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

    #for k in tqdm.tqdm(range(folds)):
        #print(f"Fold {k}")
    qrs = []
    
    #### TEST Quantile Regression on Data = Bkg. + Injected Signal ######    
    ### DATA here does not really mean data, but just a realistic form of MC simulations with 99% QCD events and very few injected signal events #####
    with h5py.File(f"{case_qr_data_injected}/{signal_folder}/data_{signal_name}_{siginj}.h5", "r") as data_f:
        branch_names = data_f['eventFeatureNames'][()].astype(str)
        #print(branch_names)
        features = data_f['eventFeatures'][()]
        mask = features[:,0]<9000
        features = features[mask]
        mjj = np.asarray(features[:,0])
        loss_indices = get_loss_indices(branch_names)

        loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 
        yss=discriminator.predict(mjj,flatten=False)
        import pdb; pdb.set_trace()
        print("Prediction complete. Now sorting data into quantiles.")

        for j,q in enumerate(quantiles):
            loss_cut_value = yss[:,j+1] # To exclude the Q-15 quantile 
            print(f'{j},{q}')
            if q == '30':
                if len(data_sel_q30) == 0:
                    data_sel_q30 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q30 = np.concatenate((data_sel_q30,np.expand_dims((loss > loss_cut_value),axis=-1)))
            if q == '70':
                if len(data_sel_q70) == 0:
                    data_sel_q70 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q70 = np.concatenate((data_sel_q70,np.expand_dims((loss > loss_cut_value),axis=-1)))
            if q == '50':
                if len(data_sel_q50) == 0:
                    data_sel_q50 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q50 = np.concatenate((data_sel_q50,np.expand_dims((loss > loss_cut_value),axis=-1)))
            if q == '10':
                if len(data_sel_q10) == 0:
                    data_sel_q10 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q10 = np.concatenate((data_sel_q10,np.expand_dims((loss > loss_cut_value),axis=-1)))
            if q == '05':
                if len(data_sel_q05) == 0:
                    data_sel_q05 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q05 = np.concatenate((data_sel_q05,np.expand_dims((loss > loss_cut_value),axis=-1)))
            if q == '01':
                if len(data_sel_q01) == 0:
                    data_sel_q01 = np.expand_dims((loss > loss_cut_value),axis=-1)
                else:
                    data_sel_q01 = np.concatenate((data_sel_q01,np.expand_dims((loss > loss_cut_value),axis=-1)))
        if len(data_loss) == 0:
            data_loss = np.expand_dims(loss,axis=-1)
            data_mjj = np.expand_dims(mjj,axis=-1)
        else:
            data_loss = np.concatenate((data_loss,np.expand_dims(loss,axis=-1)))
            data_mjj = np.concatenate((data_mjj,np.expand_dims(mjj,axis=-1)))
    

    ####### NOW DO THE SAME FOR SIGNAL EVENTS ###############
    # with h5py.File(f"/work/abal/CASE/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{signal_name}.h5", "r") as sig_f:
    #     branch_names = sig_f['eventFeatureNames'][()].astype(str)
    #     #print(branch_names)
    #     features = sig_f['eventFeatures'][()]
    #     mask = features[:,0]<9000
    #     features = features[mask]
    #     mjj = np.asarray(features[:,0])
    #     loss_indices = get_loss_indices(branch_names)

    #     loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

    #     for j,q in enumerate(quantiles):
    #         if q == '30':
    #             if len(sig_sel_q30) == 0:
    #                 sig_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q30 = np.concatenate((sig_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #         if q == '70':
    #             if len(sig_sel_q70) == 0:
    #                 sig_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q70 = np.concatenate((sig_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #         if q == '50':
    #             if len(sig_sel_q50) == 0:
    #                 sig_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q50 = np.concatenate((sig_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #         if q == '10':
    #             if len(sig_sel_q10) == 0:
    #                 sig_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q10 = np.concatenate((sig_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #         if q == '05':
    #             if len(sig_sel_q05) == 0:
    #                 sig_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q05 = np.concatenate((sig_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #         if q == '01':
    #             if len(sig_sel_q01) == 0:
    #                 sig_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
    #             else:
    #                 sig_sel_q01 = np.concatenate((sig_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
    #     if len(sig_loss) == 0:
    #         sig_loss = np.expand_dims(loss,axis=-1)
    #         sig_mjj = np.expand_dims(mjj,axis=-1)
    #     else:
    #         sig_loss = np.concatenate((sig_loss,np.expand_dims(loss,axis=-1)))
    #         sig_mjj = np.concatenate((sig_mjj,np.expand_dims(mjj,axis=-1)))



    if siginj == 0:
        sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    #print(all_event_features.shape)
    if siginj != 0:
        data_event_features = np.concatenate((data_mjj,data_loss,data_sel_q70,data_sel_q50,data_sel_q30,data_sel_q10,data_sel_q05,data_sel_q01),axis=-1)
        print(data_event_features.shape)
        

        
    if siginj != 0:
        data_outfilename = f'data_QR_tested_without_smoothing_{signal_name}_{siginj}'
        data_outfiledir = f'/work/abal/CASE/QR_datasets/run_{run_n}/{signal_name}'
        pathlib.Path(data_outfiledir).mkdir(parents=True, exist_ok=True)
        data_hf = h5py.File(f'{data_outfiledir}/{data_outfilename}.h5', 'w')
        
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
    

        

        



# %%
