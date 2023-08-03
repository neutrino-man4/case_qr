import tqdm
import sys
import os
#sys.path.insert(0, os.path.abspath('/eos/user/b/bmaier/.local/lib/python3.9/site-packages/'))
import h5py
import numpy as np
import argparse
import pdb
import pathlib
import pandas as pd
from math import sqrt
parser = argparse.ArgumentParser()
parser.add_argument("-r","--run",default=50005,type=int,help="Provide the new run number to identify your new results")
args = parser.parse_args()

run_n=args.run

# Define parameter variables for QR
kfolds=4
quantiles=['85','70','50','30','10','05','01']

# Define indices for event features in the jet file
MJJ,DETA,J1PT,J1ETA,J1PHI,J1M,J2PT,J2ETA,J2PHI,J2M,J3PT,J3ETA,J3PHI,J3M,J1TOTAL,J1RECO,J1KL,J2TOTAL,J2RECO,J2KL = range(20)

# Define indices for the quantiles
Q01,Q05,Q10,Q30,Q50,Q70,Q85=range(7)

# Define list containing the weights for all uncertainties, and all the variations.
uc_variations = ['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down','nominal'] # These files contain pT and mass arrays.
weights=['nom_weight', 'pdf_up', 'pdf_down', 'prefire_up', 'prefire_down', 'pileup_up', 'pileup_down', 'btag_up', 'btag_down', 
    'PS_ISR_up', 'PS_ISR_down', 'PS_FSR_up', 'PS_FSR_down', 'F_up', 'F_down', 'R_up', 'R_down', 'RF_up', 'RF_down', 'top_ptrw_up', 'top_ptrw_down']

uncertainties = ['JES','JER','JMS','JMR','pdf','prefire','pileup','btag','PS_ISR','PS_FSR','F','R','RF','top_ptrw']

# Define paths for reading input files
case_poly_dir = f'/work/abal/CASE/QR_models/run_{run_n}/models_lmfit_csv/unblind_data_SR/9'
case_reco_dir = f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}'

# Define paths for writing output
uc_output_dir = f'/work/abal/CASE/QR_uncertainties/run_{run_n}'
pathlib.Path(uc_output_dir).mkdir(parents=True,exist_ok=True)

# Define lists with uncertainties
# Note: In the nominal files, weight[0] is the nominal weight and weights[1:] are multiplicative factors, to get the final weight for a given correction -
# Multiply: weight[0]*weight[i>0]

# Define test samples
test_samples_X_TO_YY  = ['XToYYprimeTo4Q_MX2000_MY170_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX2000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY400_MYprime80_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime400_narrow',
                'XToYYprimeTo4Q_MX2000_MY80_MYprime80_narrow',
                'XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow',
                'XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow',
                #'XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow',
                'XToYYprimeTo4Q_MX3000_MY80_MYprime25_narrow',
                'XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY170_MYprime80_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime170_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow',
                'XToYYprimeTo4Q_MX5000_MY80_MYprime25_narrow',
                'XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow'
  
]
test_samples_Qstar_RS_W = [
                'QstarToQW_M_2000_mW_170',
                'QstarToQW_M_2000_mW_25',
                'QstarToQW_M_2000_mW_400',
                'QstarToQW_M_2000_mW_80',
                'QstarToQW_M_3000_mW_170',
                'QstarToQW_M_3000_mW_25',
                'QstarToQW_M_3000_mW_400',
                'QstarToQW_M_3000_mW_80',
                'QstarToQW_M_5000_mW_170',
                'QstarToQW_M_5000_mW_25',
                'QstarToQW_M_5000_mW_400',
                'QstarToQW_M_5000_mW_80',    
                # 'RSGravitonToGluonGluon_kMpl01_M_1000',
                # 'RSGravitonToGluonGluon_kMpl01_M_2000',
                # 'RSGravitonToGluonGluon_kMpl01_M_3000',
                # 'RSGravitonToGluonGluon_kMpl01_M_5000',
                'WkkToWRadionToWWW_M2000_Mr170',
                'WkkToWRadionToWWW_M2000_Mr400',
                'WkkToWRadionToWWW_M3000_Mr170',
                'WkkToWRadionToWWW_M3000_Mr400',
                'WkkToWRadionToWWW_M5000_Mr170',
                'WkkToWRadionToWWW_M5000_Mr400',
                'WpToBpT_Wp2000_Bp170_Top170_Zbt',
                'WpToBpT_Wp2000_Bp25_Top170_Zbt',
                'WpToBpT_Wp2000_Bp400_Top170_Zbt',
                'WpToBpT_Wp2000_Bp80_Top170_Zbt',
                'WpToBpT_Wp3000_Bp170_Top170_Zbt',
                'WpToBpT_Wp3000_Bp25_Top170_Zbt',
                'WpToBpT_Wp3000_Bp400_Top170_Zbt',
                'WpToBpT_Wp3000_Bp80_Top170_Zbt',
                'WpToBpT_Wp5000_Bp170_Top170_Zbt',
                'WpToBpT_Wp5000_Bp25_Top170_Zbt',
                'WpToBpT_Wp5000_Bp400_Top170_Zbt',
                'WpToBpT_Wp5000_Bp80_Top170_Zbt',
                ]


signals = ['XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_RECO']
signals = test_samples_Qstar_RS_W + test_samples_X_TO_YY

diffs = []    
for sig_id in tqdm.tqdm(signals):
    sig_file=sig_id+'.h5'
    signal_mjj={} # Dict to store signal feature arrays for all scales and the nominal 
    signal_loss={}
    sig_histograms={}
    num_events={}
    eff={}
    nom_eff={}
    SF_lund={}
    num_events_total={}
    diff_max={'signal_name':sig_id}
    corrdiff_max={'signal_name':sig_id}
    SF_max={'signal_name':sig_id}
    for uc in uc_variations:
        with h5py.File(os.path.join(case_reco_dir,sig_id+'_RECO',uc,sig_file),'r') as f:
            signal_jet_kinematics=f['eventFeatures'][()]
            if uc=='nominal':
                signal_uc_weights=f['sys_weights'][()]
                lund_weights=f['lund_weights'][()]
                lund_weights_sys_var=f['lund_weights_sys_var'][()]*lund_weights
                lund_weights_stat_var=f['lund_weights_stat_var'][()]
                lund_weights_pt_var=f['lund_weights_pt_var'][()]
                lund_weights_matching_unc=f['lund_weights_matching_unc'][()][0]
                
            signal_loss[uc] = np.minimum(signal_jet_kinematics[:,J1RECO]+0.5*signal_jet_kinematics[:,J1KL],signal_jet_kinematics[:,J2RECO]+0.5*signal_jet_kinematics[:,J2KL])
            signal_mjj[uc] = signal_jet_kinematics[:,MJJ]
    
    # Correct the uncertainty weights by multiplying the factor with the nominal weight. 
    for i in range(1,signal_uc_weights.shape[1]):
            signal_uc_weights[:,i]=signal_uc_weights[:,0]*signal_uc_weights[:,i]
    
    sig_qrs={}
    for q in quantiles:
        sig_tmpdf = pd.read_csv(os.path.join(case_poly_dir,f'sig_lmfit_modelresult_quant_q{q}.csv'))
    
        sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
        sig_qrs[q]=sig_p
    
    for uc in uc_variations:
        mask=signal_loss[uc]>sig_qrs['10'](signal_mjj[uc]) # For Q90
        if uc!='nominal':
            eff[uc]=np.average(mask,weights=np.ones_like(signal_mjj[uc]))

        else:
            for id,w in enumerate(weights):
                eff[w]=np.average(mask,weights=signal_uc_weights[:,id])
                eff['unity_weight']=np.average(mask)
        # Number of surviving events is stored in the same dictionary for both the scaling and the uncertainties. 
    ############ LUND WEIGHT UNCERTAINTIES
    
    nToys=100
    nom_lund_weights=lund_weights[:,0]*np.ones_like(signal_uc_weights[:,0]) # Multiply nominal weights by lund weights 
    eff['lund']=np.average(mask,weights=nom_lund_weights)
    eff_toys=[]
    pt_eff_toys=[]
    for i in range(nToys):
        stat_eff=np.average(mask,weights=lund_weights_stat_var[:,i])
        pt_eff=np.average(mask,weights=lund_weights_pt_var[:,i])
        eff_toys.append(stat_eff)
        pt_eff_toys.append(pt_eff)
    stat_toys_mean=np.mean(eff_toys)
    stat_toys_std=np.std(eff_toys)
    pt_toys_mean=np.mean(pt_eff_toys)
    pt_toys_std=np.std(pt_eff_toys)

    ##
    SF_lund['lund_stat']=abs(stat_toys_mean-eff['lund'])/eff['nom_weight'] + stat_toys_std/eff['nom_weight']
    SF_lund['lund_pt']=abs(pt_toys_mean-eff['lund'])/eff['nom_weight'] + pt_toys_std/eff['nom_weight']
    # SF_lund['lund_stat']=abs(stat_toys_mean-eff['lund'])/eff['unity_weight'] + stat_toys_std/eff['unity_weight']
    # SF_lund['lund_pt']=abs(pt_toys_mean-eff['lund'])/eff['unity_weight'] + pt_toys_std/eff['unity_weight']
    ##
    eff['lund_sys_up']=np.average(mask,weights=lund_weights_sys_var[:,0])
    eff['lund_sys_down']=np.average(mask,weights=lund_weights_sys_var[:,1])
    eff['lund_bquark_up']=np.average(mask,weights=lund_weights_sys_var[:,2])
    eff['lund_bquark_down']=np.average(mask,weights=lund_weights_sys_var[:,3])
    SF_lund['lund_sys_up']=(eff['lund_sys_up']-eff['lund'])/eff['nom_weight']
    SF_lund['lund_sys_down']=(eff['lund_sys_down']-eff['lund'])/eff['nom_weight']
    SF_lund['lund_bquark_up']=(eff['lund_bquark_up']-eff['lund'])/eff['nom_weight']
    SF_lund['lund_bquark_down']=(eff['lund_bquark_down']-eff['lund'])/eff['nom_weight']
    ###
    # SF_lund['lund_sys_up']=(eff['lund_sys_up']-eff['lund'])/eff['unity_weight']
    # SF_lund['lund_sys_down']=(eff['lund_sys_down']-eff['lund'])/eff['unity_weight']
    # SF_lund['lund_bquark_up']=(eff['lund_bquark_up']-eff['lund'])/eff['unity_weight']
    # SF_lund['lund_bquark_down']=(eff['lund_bquark_down']-eff['lund'])/eff['unity_weight']
    ###
    SF_lund['lund_sys']=max(abs(SF_lund['lund_sys_up']),abs(SF_lund['lund_sys_down']))
    SF_lund['lund_bquark']=max(abs(SF_lund['lund_bquark_up']),abs(SF_lund['lund_bquark_down']))
    
    ############
    total=0.0
    total_without_lund=0.0
    for u in uncertainties:
        SF_up=(eff[u+'_up']-eff['nom_weight'])/eff['nom_weight'] # SF = scale factor
        SF_down=(eff[u+'_down']-eff['nom_weight'])/eff['nom_weight']
        SF_max[u]=round(max(abs(SF_down),abs(SF_up))*100,4)
    
    SF_max['lund']=np.round(eff['lund']/eff['nom_weight'],4)
    #SF_max['lund']=eff['lund']/eff['unity_weight']
    SF_max['lund_stat']=np.round(SF_lund['lund_stat']*100,4)
    SF_max['lund_pt']=np.round(SF_lund['lund_pt']*100,4)
    SF_max['lund_sys']=np.round(SF_lund['lund_sys']*100,4)
    SF_max['lund_bquark']=np.round(SF_lund['lund_bquark']*100,4)
    SF_max['lund_bad_matching_unc']=np.round(SF_max['lund']*lund_weights_matching_unc,4)

    for u in SF_max:
        if u=='signal_name': continue
        total+=SF_max[u]**2
        if "lund" not in u:
            total_without_lund+=SF_max[u]**2
    SF_max['total']=np.round(sqrt(total),4)
    SF_max['total_without_lund']=np.round(sqrt(total_without_lund),4)
    diffs.append(SF_max)
    
df=pd.DataFrame(diffs)
#pdb.set_trace()
df.to_csv(os.path.join(uc_output_dir,'uncertainties_updated_with_lund.csv'))
