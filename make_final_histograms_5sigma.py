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
import pathlib,argparse
import pandas as pd
#print(ak.__version__)
#import mplhep as hep
import json
#plt.style.use(hep.style.CMS)
import subprocess
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
    #truth_label=np.argwhere(branch_names=='truth_label')[0,0]
    loss_dict = {'j1KlLoss':j1_kl,'j1RecoLoss':j1_reco,'j2KlLoss':j2_kl,'j2RecoLoss':j2_reco}#,'truthLabel':truth_label}
    return loss_dict
    
def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))

def write_log(dir,comments):
    with open(os.path.join(dir,'log.txt'), 'w') as f:
        f.write(f'QR attempt = {comments[0]}'+'\n')
        for l in comments[1:]:
            f.write(l+'\n')

##### Define directory paths
parser = argparse.ArgumentParser()
parser.add_argument("-i","--injection",type=float,default=0.0,help="Set signal c.s to inject (default = 0.). Value = injection* 1000 fb-1")
parser.add_argument("-r","--run",default=141098,type=int,help="Provide the new run number to identify your new results")
parser.add_argument("--sample",default=None,help="Provide the sample to use (rudimentary arg)") # TODO: Make it more complete

args = parser.parse_args()

#comments = ['6','Train test split = 0.8', 'QR trained on SR data skimmed by Aritra',f'smoothing range =[{bin_low},{bin_high}]',f'VAE Run = {args.run}',f'No. of folds = {nFolds}','NOTE: EACH FOLD FURTHER SPLIT AS EARLIER FOR VALIDATION. IGNORE ATTEMPTS 2-4 NOW']
# #'XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrowReco' #'WkkToWRadionToWWW_M3000_Mr170Reco'
if args.sample=='X': 
    sample = 'XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrowReco' 
elif args.sample=='Wp': 
    sample = 'WpToBpT_Wp3000_Bp170_Top170_ZbtReco'
elif args.sample=='Wkk': 
    sample = 'WkkToWRadionToWWW_M3000_Mr400Reco'
elif args.sample=='Y': 
    sample='YtoHH_Htott_Y3000_H400Reco'
elif args.sample=='Z': 
    sample='ZpToTpTp_Zp3000_Tp400Reco'
else:
    sample = 'XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrowReco' 
    print('No input sample given,falling back to XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow')

signal_name = sample
#signal_name='WkkToWRadionToWWW_M3000_Mr400Reco'
#signal_name = 'XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrowReco'
#signal_name = 'XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrowReco' #'WkkToWRadionToWWW_M3000_Mr170Reco'
#signal_name = 'YtoHH_Htott_Y3000_H400Reco'
#signal_name = 'ZpToTpTp_Zp3000_Tp400Reco'

signal_injections = [args.injection]


folds = 20
qcd_sample = 'qcdSROzData'
run_n = args.run
##### Define directory paths
case_qr_results = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}'
case_qr_models = '/work/abal/CASE/QR_models'

####### FOR LOGGING PURPOSES ONLY ##########
nAttempt=1
bin_low = 1460 
bin_high = 6800
#####

for siginj in signal_injections:
    print(f'Analysing data with {siginj} injected events of signal {signal_name}$')
    
    quantiles = ['70', '50', '30', '10', '05','01']
    all_mjj = []
    all_loss = []
    all_sel_q70 = []
    all_sel_q50 = []
    all_sel_q30 = []
    all_sel_q10 = []
    all_sel_q05 = []
    all_sel_q01 = []
    all_labels = []
    all_event_features = []
    all_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99','truth_label']

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
                tmpdf = pd.read_csv(f'{case_qr_models}/run_{run_n}/models_lmfit_csv/QCD_5sigma/{nAttempt}/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'{case_qr_models}/run_{run_n}/models_lmfit_csv/QCD_5sigma/{nAttempt}/{signal_name}/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            else:
                tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv/QCD_5sigma/{signal_name}/{str(siginj)}/{nAttempt}/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv/QCD_5sigma/{signal_name}/{str(siginj)}/{nAttempt}/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
            #p = np.poly1d(tmpdf['par'].values.tolist())

            #print("Quantile", q)
            #print(tmpdf['par'].values.tolist()[::-1])
            #print(p(1200))
            qrs.append(p)
            
        #with h5py.File(f"{case_qr_results}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}/sig_{qcd_sample}_fold_{k}.h5", "r") as bkg_f:
        with h5py.File(f"{case_qr_results}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}/data_{signal_name}_{siginj}_fold_{k}.h5", "r") as bkg_f:
            branch_names = bkg_f['eventFeatureNames'][()].astype(str)
            #print(branch_names)
            features = bkg_f['eventFeatures'][()]
            mask = features[:,0]<9000
            features = features[mask]
            mjj = np.asarray(features[:,0])
            loss_indices = get_loss_indices(branch_names)
            loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 
            #import pdb;pdb.set_trace()
            truth_labels=np.asarray(features[:,20])
            
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
                all_labels = np.expand_dims(truth_labels,axis=-1)
                
            else:
                all_loss = np.concatenate((all_loss,np.expand_dims(loss,axis=-1)))
                all_mjj = np.concatenate((all_mjj,np.expand_dims(mjj,axis=-1)))
                all_labels = np.concatenate((all_labels,np.expand_dims(truth_labels,axis=-1)))

        #if siginj == 0.:
        if k == 0:
            with h5py.File(f"{case_qr_results}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}/{signal_name}.h5", "r") as sig_f:
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



    all_event_features = np.concatenate((all_mjj,all_loss,all_sel_q70,all_sel_q50,all_sel_q30,all_sel_q10,all_sel_q05,all_sel_q01,all_labels),axis=-1)
    #if siginj == 0:
    
    sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    
    print(all_event_features.shape)
    
    case_qr_datasets=f'/ceph/abal/CASE/QR_datasets/run_{run_n}/SRData_5sigma/{nAttempt}/{signal_name}/{str(siginj)}'
    pathlib.Path(case_qr_datasets).mkdir(parents=True,exist_ok=True)
    print(f'Output datasets to be used for fitting are located at {case_qr_datasets}')
    outfilename = 'bkg'
    dt = h5py.special_dtype(vlen=str)


    if siginj != 0:
        outfilename = 'bkg+inj_%s_%s'%(signal_name,str(siginj))

    hf = h5py.File(os.path.join(case_qr_datasets,f'{outfilename}.h5'), 'w')
    #write_log(case_qr_datasets,comments)
    log_command = f'cp -r {case_qr_models}/run_{run_n}/models_lmfit_csv/QCD_5sigma/{nAttempt}/{signal_name}_{str(siginj)}/log.txt {case_qr_datasets}/'
    subprocess.call(log_command,shell=True)
    hf.create_dataset('mjj', data=np.array(all_mjj))
    hf.create_dataset('truth_label', data=np.array(all_labels))
    hf.create_dataset('loss', data=np.array(all_loss))
    hf.create_dataset('sel_q30', data=np.array(all_sel_q70))
    hf.create_dataset('sel_q50', data=np.array(all_sel_q50))
    hf.create_dataset('sel_q70', data=np.array(all_sel_q30))
    hf.create_dataset('sel_q90', data=np.array(all_sel_q10))
    hf.create_dataset('sel_q95', data=np.array(all_sel_q05))
    hf.create_dataset('sel_q99', data=np.array(all_sel_q01))
    #import pdb;pdb.set_trace()
    hf.create_dataset('eventFeatures', data=np.array(all_event_features))
    hf.create_dataset('eventFeatureNames', data=np.array(all_event_feature_names_reversed,dtype=dt))
    hf.close()
    #if siginj == 0:
    sig_outfilename = 'signal_%s'%(signal_name)
    sig_hf = h5py.File(os.path.join(case_qr_datasets,f'{sig_outfilename}.h5'), 'w')
    sig_hf.create_dataset('mjj', data=np.array(sig_mjj))
    sig_hf.create_dataset('loss', data=np.array(sig_loss))
    sig_hf.create_dataset('sel_q30', data=np.array(sig_sel_q70))
    sig_hf.create_dataset('sel_q50', data=np.array(sig_sel_q50))
    sig_hf.create_dataset('sel_q70', data=np.array(sig_sel_q30))
    sig_hf.create_dataset('sel_q90', data=np.array(sig_sel_q10))
    sig_hf.create_dataset('sel_q95', data=np.array(sig_sel_q05))
    sig_hf.create_dataset('sel_q99', data=np.array(sig_sel_q01))
    sig_hf.create_dataset('eventFeatures', data=np.array(sig_event_features))
    sig_hf.create_dataset('eventFeatureNames', data=np.array(sig_event_feature_names_reversed,dtype=dt))
    sig_hf.close()
        
    


