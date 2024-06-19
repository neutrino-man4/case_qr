import h5py
import argparse
import os
import numpy as np
import pathlib
import pdb
nAttempt = 5
run=50005
case_dataset_path=f'/ceph/abal/CASE/QR_datasets/run_{run}/QCD_MC_for_asimov/{nAttempt}'
#sig_name='WpToBpT_Wp3000_Bp400_Top170_Zbt'
sig_name = 'XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow' #'XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrowReco' #'WkkToWRadionToWWW_M3000_Mr170Reco'
#output_dir=os.path.join(case_dataset_path,'signal_injection_XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrowReco')
# seed='random-1'
# output_dir=os.path.join(case_dataset_path,'seed_'+seed)

# lumi=137.0
# sig_incl=150000

injections= [1700,1500,1300, 1100, 900, 700, 500, 300]
injections= [720,600,480,360,240]
injections={0:280,5:450,10:585,20:970,30:1170,40:1800}


bkg_file=h5py.File(os.path.join(case_dataset_path,'bkg.h5'),'r')

for s in [0,5,10,20,30,40]:
    eventsB=bkg_file['eventFeatures'][()]
    if s==0:
        sig_file=h5py.File(os.path.join(case_dataset_path,f'signal_{sig_name}Reco.h5'),'r')
    else:
        sig_file=h5py.File(os.path.join(case_dataset_path,f'signal_{sig_name}_smear{s}Reco.h5'),'r')
    
    nested_dir=f'smear_{s}'
    if s==0:
        nested_dir='narrow'
        
    output_dir=os.path.join(case_dataset_path,nested_dir)

    eventsS=sig_file['eventFeatures'][()]

    event_index_end=eventsB.shape[1]

    feature_names=bkg_file['eventFeatureNames'][()]
    dt = h5py.special_dtype(vlen=str)

    # Create an array for the truth labels
    sig_truth=np.ones_like(sig_file['loss'][()])
    bkg_truth=np.zeros_like(bkg_file['loss'][()])

    sig_num=sig_truth.shape[0]

    for k in bkg_file.keys():
        if 'eventFeature' in k: continue
        eventsB=np.hstack((eventsB,bkg_file[k][()]))
        eventsS=np.hstack((eventsS,sig_file[k][()]))
    pathlib.Path(output_dir).mkdir(parents=True,exist_ok=True)

    eventsB=np.hstack((eventsB,bkg_truth))
    eventsS=np.hstack((eventsS,sig_truth))
    new_keys=list(bkg_file.keys())+['truth_label']
    #for sig_scale in injections:
    sig_scale=injections[s]
    rng = np.random.default_rng()
    #xsec=inj*1000. # Convert to fb-1
    #sig_scale=int((xsec*lumi*sig_num)/sig_incl)
    print(f'injecting {sig_scale} events')
    injected_sig=rng.choice(eventsS,size=sig_scale,replace=False)
    
    data_arr=np.concatenate((eventsB,injected_sig))
    
    #inj_hf=h5py.File(os.path.join(output_dir,f'bkg+inj_{sig_name}_{sig_scale}.h5'),'w')
    #injOnly_hf=h5py.File(os.path.join(output_dir,f'only_inj_{sig_name}_{sig_scale}.h5'),'w')
    inj_hf=h5py.File(os.path.join(output_dir,f'bkg+inj_{sig_name}.h5'),'w')
    injOnly_hf=h5py.File(os.path.join(output_dir,f'only_inj_{sig_name}.h5'),'w')
    
    for i,k in enumerate(new_keys):
        if k=='eventFeatureNames':
            inj_hf.create_dataset('eventFeatureNames', data=np.array(feature_names,dtype=dt))
            injOnly_hf.create_dataset('eventFeatureNames', data=np.array(feature_names,dtype=dt))
        elif k=='eventFeatures':
            inj_hf.create_dataset('eventFeatures',data=data_arr[:,:event_index_end])
            injOnly_hf.create_dataset('eventFeatures',data=injected_sig[:,:event_index_end])
        else:
            inj_hf.create_dataset(k,data=np.expand_dims(data_arr[:,event_index_end+i-2],axis=-1)) # To account for the first two keys: eventFeatureNames and eventFeatures            
            injOnly_hf.create_dataset(k,data=np.expand_dims(injected_sig[:,event_index_end+i-2],axis=-1)) # To account for the first two keys: eventFeatureNames and eventFeatures            

    inj_hf.close()
    injOnly_hf.close()
