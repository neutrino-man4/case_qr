import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import setGPU
import numpy as np
import random
import tensorflow as tf
from recordtype import recordtype
import pathlib
import copy
import sys
import json
import pandas as pd
import pdb
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
import case_analysis.util.sample_names as samp
import case_paths.phase_space.cut_constants as cuts
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import argparse

def get_generated_events(filename):

   with open('files_count.json') as f:
      data = json.load(f)

   N = 0
   found = False
   for k in data.keys():
      if k in filename or k.replace('_EXT','') in filename:
         N += data[k][0]
         found = True

   if not found:
      print ( "######### no matching key in files_count.json for file "+filename+", EXIT !!!!!!!!!!!!!!")
      sys.exit()

   if not found:
      print ( "######### no matching key in files_count.json for file "+filename+", EXIT !!!!!!!!!!!!!!")
      sys.exit()

   print ( " in get_generated_events N = ",N)
   return N


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))

def fitted_selection(sample, strategy_id, polynomial):
    loss_strategy = lost.loss_strategy_dict[strategy_id]
    loss = loss_strategy(sample)
    loss_cut = polynomial
    return loss > loss_cut(sample['mJJ'])


#set_seeds(777)

#****************************************#
#           set runtime params
#****************************************#


#parser = argparse.ArgumentParser()
#parser.add_argument("-i","--injection",type=float,default=0.0,help="Set signal lumi to inject (default = 0.). Value = injection* 1000 fb-1")
#args = parser.parse_args()

#sample = sys.argv[1] #'grav_3p5_narrow'
#mass = float(sys.argv[2])
#inj = float(sys.argv[3])
sample = 'WkkToWRadionToWWW_M3000_Mr170Reco'
mass = 3000.
injected=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

print(injected)


signals = samp.all_samples
masses = [mass]


#quantiles = [0.5]

# to run
Parameters = recordtype('Parameters','run_n, qcd_sample_id, sig_sample_id, strategy_id, injected_sample_id, read_n')
params = Parameters(run_n=28332,
                    qcd_sample_id='qcdSigMCOrigReco',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    injected_sample_id='qcdSigMCOrigReco_injectedJets',
                    read_n=int(1e8))


#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})


print("Lumi calculation: ...")


#qcd_gen_events = get_generated_events(params.qcd_sample_id)
#sig_gen_events = get_generated_events(options.sigFile)
#qcd_xsec = 8.73e6 # [fb]                                 
#lumi = qcd_gen_events/qcd_xsec
#lumi=64.1
#lumi=56.4
lumi=26.8
print("Lumi: ", lumi)
print("Lumi calculation done")


qcd_sample= dapr.make_qcd_train_test_datasets(params, paths,which_fold=-1, nfold=-1, train_split=0.,**cuts.signalregion_cuts)

print('Read in QCD dataset \n ______________')

for sig_sample_id in signals:
    for inj in injected:

        print(inj)
        
        params.sig_sample_id = sig_sample_id
        sig_sample_ini = js.JetSample.from_input_dir(params.sig_sample_id, paths.sample_dir_path(params.sig_sample_id), **cuts.signalregion_cuts)
        
        # ************************************************************
        #     for each signal xsec: train and apply QR
        # ************************************************************
        
        
        param_dict = {'$sig_name$': params.sig_sample_id, '$loss_strat$': params.strategy_id}
        experiment = ex.Experiment(run_n=params.run_n, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
        result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), **param_dict}) # in selection paths new format with run_x, sig_x, ...
        
        # ************************************************************
        #                     train
        # ************************************************************
        
        # create new test samples for new xsec QR (quantile cut results)
        #qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
        
        sig_sample = copy.deepcopy(sig_sample_ini)


        print("Selected QCD events: ", params.qcd_sample_id)
        signal_events_inclusive = 150000#float(get_generated_events(params.sig_sample_id))
        signal_events_selected = float(len(sig_sample_ini))
        exp_incl_events = float(inj*1000)*float(lumi)
        exp_sel_events = exp_incl_events * signal_events_selected/signal_events_inclusive
        how_much_to_inject = int(exp_sel_events)# /params.kfold)
        print("%%%%%%%%%%%%%%%%%%%%")
        print("Injecting %i signal events"%how_much_to_inject)
        print("%%%%%%%%%%%%%%%%%%%%")

        if how_much_to_inject == 0:
            mixed_sample = dapr.inject_signal(qcd_sample, sig_sample_ini, how_much_to_inject, train_split = 0.)
        else:
            mixed_sample = dapr.inject_signal(qcd_sample, sig_sample_ini, how_much_to_inject, train_split = 0.)
            #injected_signal_samples.append(injected_signal)
        
        # Keep a copy for debugging        
        mixed_sample_original=copy.deepcopy(mixed_sample)
        #pdb.set_trace()
        
        mixed_sample.dump(paths.sample_file_path(params.injected_sample_id, additional_id=sig_sample_id, mkdir=True,overwrite='True',customname=f'qcd_sig_orig_reco_injected_{sig_sample_id}_{inj}'))

#print("Dictionary:")
#print(discriminators)
