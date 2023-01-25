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

import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr
import pofah.path_constants.sample_dict_file_parts_selected as sdfs
import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import dadrah.selection.qr_workflow as qrwf
import analysis.analysis_discriminator as andi
import dadrah.util.data_processing as dapr
import pofah.phase_space.cut_constants as cuts

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


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

iteration = sys.argv[1]
sample = sys.argv[2] #'grav_3p5_narrow'
mass = float(sys.argv[3])
inj = float(sys.argv[4])

signal_contamin = { ('na', 0): [[0]]*4,
                    ('na', 100): [[1061], [1100], [1123], [1140]], # narrow signal. number of signal contamination; len(sig_in_training_nums) == len(signals)
                    ('br', 0): [[0]]*4,
                    ('br', 100): [[1065], [1094], [1113], [1125]], # broad signal. number of signal contamination; len(sig_in_training_nums) == len(signals)
                }


#bin_edges = np.array([1457, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 2111, 2206,
#                        2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093, 3221, 3353, 3490, 3632, 3778, 3928,
#                        4084, 4245, 4411, 4583, 4760, 4943, 5132, 5327, 5574, 5737, 5951]).astype('float')

#bin_centers = [(high+low)/2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]


# signals
resonance = 'na'
signals = [sample]#,'WkkToWRadionToWWW_M3000_Mr170Reco']
masses = [mass]
xsecs = [0.]
sig_in_training_nums_arr = signal_contamin[(resonance, xsecs[0])] # TODO: adapt to multiple xsecs

# quantiles
quantiles = [0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
#quantiles = [0.1, 0.3]
regions = ["A","B","C","D","E"]
#quantiles = [0.5]

# to run
Parameters = recordtype('Parameters','run_n, qcd_sample_id, sig_sample_id, strategy_id, epochs, kfold, poly_order, read_n')
params = Parameters(run_n=50000,
                    qcd_sample_id='qcd',
                    sig_sample_id=None, # set sig id later in loop
                    strategy_id='rk5_05',
                    epochs=800,
                    kfold=4,
                    #kfold=3,
                    poly_order=6,
                    read_n=int(1e8))





#result_dir = '/data/t3home000/bmaier/CASE/QR_results/analysis/vqr_run_%s/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/' % str(params.run_n) + '/maurizio_envelope'

#subprocess.call("mkdir -p %s"%result_dir,shell=True)

#****************************************#
#           read in qcd data
#****************************************#
paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(params.run_n)})


print("Lumi calculation: ...")


qcd_gen_events = get_generated_events(params.qcd_sample_id)
#sig_gen_events = get_generated_events(options.sigFile)
qcd_xsec = 8.73e6 # [fb]                                 
lumi = qcd_gen_events/qcd_xsec
#lumi=64.1
#lumi=56.4
lumi=64
print("Lumi: ", lumi)
print("Lumi calculation done")

#exit(1)



cut_results = {}
polynomials = {}
spolynomials = {}
discriminators = {}

all_coeffs = {}

nosiginj_chunks = []
chunks = []
signal_samples = []


for k in range(params.kfold):

    # if datasets not yet prepared, prepare them, dump and return (same qcd train and testsample for all signals and all xsecs)
    qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, which_fold=k, nfold=params.kfold, **cuts.signalregion_cuts)
    nosiginj_chunks.append(qcd_train_sample)


for k in range(params.kfold):

    # if datasets not yet prepared, prepare them, dump and return (same qcd train and testsample for all signals and all xsecs)

    #qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, which_fold=k, nfold=params.kfold, **cuts.signalregion_Uppercuts)
    qcd_train_sample, qcd_test_sample_ini = dapr.make_qcd_train_test_datasets(params, paths, which_fold=k, nfold=params.kfold, **cuts.signalregion_cuts)

    # test sample corresponds to the other N-1 folds. It will not be used in the following.
    
    #****************************************#
    #      for each signal: QR
    #****************************************#
    
    for sig_sample_id, sig_in_training_nums, mass in zip(signals, sig_in_training_nums_arr, masses):
        
        params.sig_sample_id = sig_sample_id
        sig_sample_ini = js.JetSample.from_input_dir(params.sig_sample_id, paths.sample_dir_path(params.sig_sample_id), **cuts.signalregion_cuts)
        
        # ************************************************************
        #     for each signal xsec: train and apply QR
        # ************************************************************
        
        for xsec, sig_in_training_num in zip(xsecs, sig_in_training_nums):
            
            param_dict = {'$sig_name$': params.sig_sample_id, '$sig_xsec$': str(int(xsec)), '$loss_strat$': params.strategy_id}
            experiment = ex.Experiment(run_n=params.run_n, param_dict=param_dict).setup(model_dir_qr=True, analysis_dir_qr=True)
            result_paths = sf.SamplePathDirFactory(sdfs.path_dict).update_base_path({'$run$': str(params.run_n), **param_dict}) # in selection paths new format with run_x, sig_x, ...
            
            # ************************************************************
            #                     train
            # ************************************************************
            
            # create new test samples for new xsec QR (quantile cut results)
            #qcd_test_sample = copy.deepcopy(qcd_test_sample_ini)
            
            sig_sample = copy.deepcopy(sig_sample_ini)


            print("Selected QCD events: ", params.qcd_sample_id)
            signal_events_inclusive = float(get_generated_events(params.sig_sample_id))
            signal_events_selected = float(len(sig_sample_ini))
            exp_incl_events = float(inj*1000)*float(lumi)
            exp_sel_events = exp_incl_events * signal_events_selected/signal_events_inclusive
            how_much_to_inject = int(exp_sel_events/params.kfold)
            print("%%%%%%%%%%%%%%%%%%%%")
            print("Injecting %i signal events"%how_much_to_inject)
            print("%%%%%%%%%%%%%%%%%%%%")


            mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, how_much_to_inject, train_split = 0.66)
            #mixed_train_sample, mixed_valid_sample = dapr.inject_signal(qcd_train_sample, sig_sample_ini, inj, train_split = 0.66)
            
            if k == 0:
                #sig_sample.dump(result_paths.sample_file_path(params.sig_sample_id))
                signal_samples.append(sig_sample)

            chunks.append(mixed_train_sample.merge(mixed_valid_sample))
                
            # train QR model
            discriminator = qrwf.train_VQRv1(quantiles, mixed_train_sample, mixed_valid_sample, params)

            discriminators.update({"fold_%s"%str(k):discriminator})

#print("Dictionary:")
#print(discriminators)

bin_low = 1100
bin_high = 8010

bins = np.linspace(bin_low,bin_high,bin_high-bin_low)

bin_centers = [(high+low)/2 for low, high in zip(bins[:-1], bins[1:])]

#bin_centers = [1200, 1300, 1474.1252, 1560.6403, 1694.2654, 1827.9368, 1961.5662, 2095.2969, 2228.7554, 2362.04, 2495.531, 2629.0693, 2762.6633, 2895.8464, 3030.178, 3163.2517, 3296.309, 3429.8882, 3563.1992, 3697.7837, 3828.7358, 3961.4727, 4099.5864, 4227.971, 4365.957, 4494.6973, 4632.1885, 4764.8906, 4893.6655, 5024.822]

cut_dict = {}

for k in range(0,params.kfold):
    for q,quantile in enumerate(quantiles):
        inv_quant = round((1.-quantile),2)
        qrcuts = np.empty([0, len(bin_centers)])
        counter = 0 
        for l in range(0,params.kfold):
            if k == l:
                continue

            yss = discriminators["fold_%s"%str(l)].predict(bin_centers)
            
            split_yss = np.array(np.split(np.array(yss),len(bin_centers)),dtype=object)[:,q]
            qrcuts = np.append(qrcuts, split_yss[np.newaxis,:], axis=0) 
        
        y_mean = np.mean(qrcuts,axis=0)

        from lmfit.models import PolynomialModel
        from lmfit.model import save_model, save_modelresult
        
        model = PolynomialModel(degree=params.poly_order)
        x_model = []
        y_model = []
        w_model = []
        for i in range(len(qrcuts)):
            for j in range(len(qrcuts[0])):
                x_model.append(bin_centers[j])

                y_model.append(qrcuts[i][j])
                w_model.append((1./500)*np.exp(-(bin_centers[j]-1450)/500.))
                w_model.append(1.)

        pars = model.guess(y_model, x=x_model)

        out = model.fit(y_model, pars, weights=w_model, x=x_model)

        pars = []
        parserr = []
        
        for key in out.params:
            print(key, "=", out.params[key].value, "+/-", out.params[key].stderr)
            pars.append(out.params[key].value)
            parserr.append(out.params[key].stderr)

        #save_model(out, 'models_lmfit_csv/bkg_lmfit_modelresult_fold_{}_quant_q{:02}.sav'.format(str(k),int(inv_quant*100)))

        data = {'par':pars,'err':parserr}
        df = pd.DataFrame.from_dict(data)

        if inj == 0:
            subprocess.call('mkdir -p models_lmfit_csv',shell=True)
            df.to_csv('models_lmfit_csv/bkg_lmfit_modelresult_fold_{}_quant_q{:02}.csv'.format(str(k),int(inv_quant*100)))
        else:
            subprocess.call('mkdir -p models_lmfit_csv_{}_{}'.format(sample,inj),shell=True)
            df.to_csv('models_lmfit_csv_{}_{}/bkg_lmfit_modelresult_fold_{}_quant_q{:02}.csv'.format(sample,inj,str(k),int(inv_quant*100)))

        cut_dict['{}_q{:02}'.format(str(k),int(inv_quant*100))] = y_mean

    if inj == 0:
        nosiginj_chunks[k].dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True),fold=k)
    else:
        chunks[k].dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True, overwrite=True, customname='data_{}_{}'.format(sample,inj)),fold=k)


if inj == 0:
    final_bkgsample = nosiginj_chunks[0]
    for k in range(1,params.kfold):
        final_bkgsample = final_bkgsample.merge(nosiginj_chunks[k])
    final_bkgsample.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True))
else:
    final_datasample = chunks[0]
    for k in range(1,params.kfold):
        final_datasample = final_datasample.merge(chunks[k])
    final_datasample.dump(result_paths.sample_file_path(params.qcd_sample_id, mkdir=True, overwrite=True, customname='data_{}_{}'.format(sample,inj)))

sig_cut_dict = {}

for q,quantile in enumerate(quantiles):
    inv_quant = round((1.-quantile),2)
    qrcuts = np.empty([0, len(bin_centers)])
    counter = 0 
    for l in range(0,params.kfold):
        #if k == l:
        #    continue
        
        yss = discriminators["fold_%s"%str(l)].predict(bin_centers)
        
        split_yss = np.array(np.split(np.array(yss),len(bin_centers)),dtype=object)[:,q]
        qrcuts = np.append(qrcuts, split_yss[np.newaxis,:], axis=0) 
        
    y_mean = np.mean(qrcuts,axis=0)

    from lmfit.models import PolynomialModel
    from lmfit.model import save_model, save_modelresult
    
    model = PolynomialModel(degree=params.poly_order)
    x_model = []
    y_model = []
    w_model = []
    for i in range(len(qrcuts)):
        for j in range(len(qrcuts[0])):
            x_model.append(bin_centers[j])            
            y_model.append(qrcuts[i][j])
            w_model.append((1./500)*np.exp(-(bin_centers[j]-1450)/500.))
            #w_model.append(1)

    pars = model.guess(y_model, x=x_model)

    out = model.fit(y_model, pars, weights=w_model, x=x_model)

    pars = []
    parserr = []
    
    for key in out.params:
        #print(key, "=", out.params[key].value, "+/-", out.params[key].stderr)
        pars.append(out.params[key].value)
        parserr.append(out.params[key].stderr)

    #save_model(out, 'models_lmfit_csv/bkg_lmfit_modelresult_fold_{}_quant_q{:02}.sav'.format(str(k),int(inv_quant*100)))
    
    data = {'par':pars,'err':parserr}
    df = pd.DataFrame.from_dict(data)
    if inj == 0:
        df.to_csv('models_lmfit_csv/sig_lmfit_modelresult_quant_q{:02}.csv'.format(int(inv_quant*100)))
    else:
        df.to_csv('models_lmfit_csv_{}_{}/sig_lmfit_modelresult_quant_q{:02}.csv'.format(sample,inj,int(inv_quant*100)))
        for i,s in enumerate(signal_samples):
            s.dump(result_paths.sample_file_path(params.sig_sample_id))

