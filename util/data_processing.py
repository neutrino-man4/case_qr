import os
import case_paths.jet_sample as js
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import gc
import pdb
from tqdm import tqdm
import time

def merge_qcd_base_and_ext_datasets(params, paths, **cuts):
    # read qcd & qcd ext
    qcd_sr_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=params.read_n, **cuts) 
    qcd_sr_ext_sample = js.JetSample.from_input_dir(params.qcd_ext_sample_id, paths.sample_dir_path(params.qcd_ext_sample_id), read_n=params.read_n, **cuts)
    # merge to combined jet sample and split into training and test parts
    return qcd_sr_sample.merge(qcd_sr_ext_sample) 


def make_qcd_train_test_datasets(params, paths, which_fold=-1, nfold=-1, train_split=0.2, **cuts):

    qcd_sr_sample = js.JetSample.from_input_dir(params.qcd_sample_id, paths.sample_dir_path(params.qcd_sample_id), read_n=params.read_n, **cuts) 
    
    if train_split==0.:
        return qcd_sr_sample
    
    if nfold < 0:
        qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sr_sample, train_split, new_names=(params.qcd_sample_id+'_train', params.qcd_sample_id+'_test'), which_fold=which_fold, nfold=nfold)
    else:
        qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sr_sample, train_split, new_names=(params.qcd_sample_id+'_train_fold_%s'%str(which_fold), params.qcd_sample_id+'_test_fold_%s'%(which_fold)), which_fold=which_fold, nfold=nfold)

    
    # if nfold < 0:
    # #    write to file
    #    qcd_train.dump(paths.sample_file_path(params.qcd_train_sample_id, mkdir=True))
    #    qcd_test.dump(paths.sample_file_path(params.qcd_test_sample_id, mkdir=True))
    
    return qcd_train, qcd_test


def inject_signal(qcd_train_sample, sig_sample, sig_in_training_num, train_split=0.75):
    if sig_in_training_num == 0:
        return js.split_jet_sample_train_test(qcd_train_sample, train_split)
    # sample random sig_in_train_num events from signal sample
    sig_train_sample = sig_sample.sample(n=sig_in_training_num)
    # merge qcd and signal
    mixed_sample = qcd_train_sample.merge(sig_train_sample)
    
    if train_split==0.:
        return mixed_sample,sig_train_sample
    
    # split training data into train and validation set
    mixed_sample_train, mixed_sample_valid = js.split_jet_sample_train_test(mixed_sample, train_split)
    return mixed_sample_train, mixed_sample_valid, sig_train_sample

def replace_jets(mixed_sample,mass_range=500.):
    jet1_columns = ['j1Pt', 'j1Eta', 'j1Phi', 'j1M','j1TotalLoss','j1RecoLoss', 'j1KlLoss']
    jet2_columns = ['j2Pt', 'j2Eta', 'j2Phi', 'j2M','j2TotalLoss','j2RecoLoss', 'j2KlLoss']

    pd.options.mode.chained_assignment = None 
    jet_samples=mixed_sample.data
    
    jet_samples.loc[:,'log_j1Pt']=jet_samples.loc[:,'j1Pt'].apply(lambda x: np.log(x))#np.log(jetset1.j1Pt)
    jet_samples.loc[:,'log_j2Pt']=jet_samples.loc[:,'j2Pt'].apply(lambda x: np.log(x))#np.log(jetset1.j1Pt)
        

    dist1=['log_j1Pt','j1Eta']
    dist2=['log_j2Pt','j2Eta']
    print(f"Total events to perform re-sampling over: {jet_samples.shape[0]}")
    binning = np.array([1450, 1529, 1604, 1681, 1761, 1844, 1930, 2019,
               2111, 2206, 2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093,
               3221, 3353, 3490, 3632, 3778, 3928, 4084, 4245, 4411, 4583,
               4760, 4943, 5132, 5327, 5527,1000000]) # Set fantastically high upper limit, faster than checking if in last bin
    
    for i,event in tqdm(jet_samples.iterrows(),total=jet_samples.shape[0]):
        
        bin_index=np.digitize(event.mJJ,binning)
        
        lower_limit_mjj = binning[bin_index-1]
        upper_limit_mjj = binning[bin_index]
        jet1 = event[['log_j1Pt','j1Pt','j1Eta']]
        jet2 = event[['log_j2Pt','j2Pt','j2Eta']]
        
        pt_window=10.
        eta_window=0.1
        
        #cut = ((jet_samples.mJJ < lower_limit_mjj) | (jet_samples.mJJ > upper_limit_mjj)) #\
        #masked=jet_samples[cut]
        
        a1=time.time()
        c=0
        while True:
            cut = ((jet_samples.mJJ < lower_limit_mjj) | (jet_samples.mJJ > upper_limit_mjj)) \
                & (abs(jet_samples.j1Pt-jet1.j1Pt)<pt_window) & (abs(jet_samples.j2Pt-jet2.j2Pt)<pt_window) \
                & (abs(jet_samples.j1Eta-jet1.j1Eta)<eta_window) & (abs(jet_samples.j2Eta-jet2.j2Eta)<eta_window) 
               
            masked=jet_samples[cut]
            if masked.shape[0]!=0: break
            c+=1
            pt_window*=10
            eta_window*=10  # If empty, enlarge window
            #del cut; gc.collect()

        
        jetset1 = masked[['log_j1Pt','j1Eta']]
        jetset2 = masked[['log_j2Pt','j2Eta']]
        
        jetset1['distances'] = cdist([jet1[dist1]],jetset1[dist1]).flatten()
        jetset2['distances'] = cdist([jet2[dist2]],jetset2[dist2]).flatten()
        
        closest_jet1_idx = jetset1['distances'].idxmin()
        closest_jet2_idx = jetset2['distances'].idxmin()
        
        
        jet_samples.loc[i,jet1_columns]=jet_samples.loc[closest_jet1_idx,jet1_columns]
        jet_samples.loc[i,jet2_columns]=jet_samples.loc[closest_jet2_idx,jet2_columns]
        
        
    jet_samples=jet_samples.drop(['log_j1Pt','log_j2Pt'],axis=1)
    return mixed_sample