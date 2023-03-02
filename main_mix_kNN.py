import argparse
import glob
import h5py
import numpy as np
import sys
import tqdm
from sklearn.neighbors import KDTree
import os
import pdb
import case_paths.util.sample_factory as sf
import case_paths.path_constants.sample_dict_file_parts_reco as sdfr
import case_analysis.util.sample_names as samp

def normalize(df,max=None):
    if max:
        return df/max
    return (df-df.min())/(df.max()-df.min())

def mask(df,q=0.9):
    return df>np.quantile(df,q=q)

def create_libraries(binning):
    reg = 250
    
    branches = ['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M',
                'j2Pt', 'j2Eta', 'j2Phi', 'j2M', 'j1RecoLoss', 'j1KlLoss',
                'j2RecoLoss', 'j2KlLoss']
    
    k_NN=5
    
    run_n=50005    
    signals=samp.all_samples
    injected=[0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
    
    #sig_sample_id=signal
        
    injected_id='qcdSigMCOrigReco_injectedJets'
    mixed_id='qcdSigMCOrigReco_mixedJets'
    paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': 'run_'+str(run_n)})

    signals = ['WkkToWRadionToWWW_M3000_Mr170Reco']
    injected=[0.1]
    
    for sig_sample_id in signals:
        print('###############################')
        print(sig_sample_id)
        print('###############################')
        
        for inj in injected:    
        
            libs_mjj = {}
            
            libs_j2pt = {}
            libs_j2eta = {}
            libs_j2phi = {}
            libs_j2reco = {}
            libs_j2kl = {}
            
            libs_j1pt = {}
            libs_j1eta = {}
            libs_j1phi = {}
            libs_j1reco = {}
            libs_j1kl = {}
            
            evts_mjj = {}
            evts_deta = {}
            evts_j1pt = {}
            evts_j1eta = {}
            evts_j1phi = {}
            evts_j1reco = {}
            evts_j1kl = {}
            evts_j2pt = {}
            evts_j2eta = {}
            evts_j2phi = {}
            evts_j2reco = {}
            evts_j2kl = {}

            
            idxs = {}

            kdtrees = {}
            kdtrees1 = {}
            #for i,filename in enumerate(glob.glob(path+"/*h5")):
            
            
            
            #filename=paths.sample_file_path(injected_id,additional_id=sig_sample_id,overwrite=True,customname=f'data_{sig_sample_id}_{inj}')
            
            filename=f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}/qcd_sig_orig_RECO_injected_jets/sig_MCOrig_QR_Reco.h5'
            
            print(f'Performing mixing on {filename}')
            with h5py.File(filename, "r") as f:
                features = f['eventFeatures'][()]
                
                # Get the indices of the featureNames when opening the first file
                if len(idxs) == 0:
                    featureNames = f['eventFeatureNames'][()].astype(str)
                    for b in branches:
                        try:
                            idxs[b] = np.argwhere(featureNames == b)[0][0]
                        except NameError:
                            print(f"Column {b} probably doesn't exist in this dataset.")
                            pass
                features[:,idxs['j1Pt']]=normalize(features[:,idxs['j1Pt']])
                features[:,idxs['j1Eta']]=normalize(features[:,idxs['j1Eta']],max=2.5)
                features[:,idxs['j1Phi']]=normalize(features[:,idxs['j1Phi']],max=3.15)
                
                features[:,idxs['j2Pt']]=normalize(features[:,idxs['j2Pt']])
                features[:,idxs['j2Eta']]=normalize(features[:,idxs['j2Eta']],max=2.5)
                features[:,idxs['j2Phi']]=normalize(features[:,idxs['j2Phi']],max=3.15)
                
                        
                for b,lower in enumerate(binning): 
                    if b ==	len(binning)-1:
                        upper = 1e9
                    else:
                        upper = binning[b+1]
                    
                    #pdb.set_trace()
                    # Sorting events according to the bins
                    tmp_features = features[ (features[:,idxs['mJJ']] > lower ) & (features[:,idxs['mJJ']] <= upper )]

                    # tmp_features contains all events in bin b

                    #print('Bin',b,'contains',len(tmp_features),'events')
                    if b not in evts_mjj:
                        evts_mjj[b] = tmp_features[:,idxs['mJJ']]
                        evts_deta[b] = tmp_features[:,idxs['DeltaEtaJJ']]
                        evts_j1pt[b] = (tmp_features[:,idxs['j1Pt']]) # WARNING: NP.LOG IF NOT NORMALIZED
                        evts_j1eta[b] = tmp_features[:,idxs['j1Eta']]
                        evts_j1phi[b] = tmp_features[:,idxs['j1Phi']]
                        evts_j1reco[b] = tmp_features[:,idxs['j1RecoLoss']]
                        evts_j1kl[b] = tmp_features[:,idxs['j1KlLoss']]
                        evts_j2pt[b] = (tmp_features[:,idxs['j2Pt']]) # WARNING: NP.LOG IF NOT NORMALIZED
                        evts_j2eta[b] = tmp_features[:,idxs['j2Eta']]
                        evts_j2phi[b] = tmp_features[:,idxs['j2Phi']]
                        evts_j2reco[b] = tmp_features[:,idxs['j2RecoLoss']]
                        evts_j2kl[b] = tmp_features[:,idxs['j2KlLoss']]
                    else:
                        evts_mjj[b] = np.concatenate((evts_mjj[b],tmp_features[:,idxs['mJJ']]),axis=0)
                        evts_deta[b] = np.concatenate((evts_deta[b],tmp_features[:,idxs['DeltaEtaJJ']]),axis=0)
                        evts_j1pt[b] = np.concatenate((evts_j1pt[b],(tmp_features[:,idxs['j1Pt']])),axis=0) # WARNING: NP.LOG IF NOT NORMALIZED
                        evts_j1eta[b] = np.concatenate((evts_j1eta[b],tmp_features[:,idxs['j1Eta']]),axis=0)
                        evts_j1phi[b] = np.concatenate((evts_j1phi[b],tmp_features[:,idxs['j1Phi']]),axis=0)
                        evts_j1reco[b] = np.concatenate((evts_j1reco[b],tmp_features[:,idxs['j1RecoLoss']]),axis=0)
                        evts_j1kl[b] = np.concatenate((evts_j1kl[b],tmp_features[:,idxs['j1KlLoss']]),axis=0)
                        evts_j2pt[b] = np.concatenate((evts_j2pt[b],(tmp_features[:,idxs['j2Pt']])),axis=0)
                        evts_j2eta[b] = np.concatenate((evts_j2eta[b],tmp_features[:,idxs['j2Eta']]),axis=0) # WARNING: NP.LOG IF NOT NORMALIZED
                        evts_j2phi[b] = np.concatenate((evts_j2phi[b],tmp_features[:,idxs['j2Phi']]),axis=0)
                        evts_j2reco[b] = np.concatenate((evts_j2reco[b],tmp_features[:,idxs['j2RecoLoss']]),axis=0)
                        evts_j2kl[b] = np.concatenate((evts_j2kl[b],tmp_features[:,idxs['j2KlLoss']]),axis=0)
                    # Creating a library per bin that corresponds to the inverse
                    tmp_features = features[ (features[:,idxs['mJJ']] <= lower ) | (features[:,idxs['mJJ']] > upper )]
                    if b not in libs_mjj:
                        libs_j2pt[b] = (tmp_features[:,idxs['j2Pt']]) # WARNING: NP.LOG IF NOT NORMALIZED
                        libs_j2eta[b] = tmp_features[:,idxs['j2Eta']]
                        libs_j2phi[b] = tmp_features[:,idxs['j2Phi']]
                        libs_j2reco[b] = tmp_features[:,idxs['j2RecoLoss']]
                        libs_j2kl[b] = tmp_features[:,idxs['j2KlLoss']]
                        
                        libs_j1pt[b] = (tmp_features[:,idxs['j1Pt']]) # WARNING: NP.LOG IF NOT NORMALIZED
                        libs_j1eta[b] = tmp_features[:,idxs['j1Eta']]
                        libs_j1phi[b] = tmp_features[:,idxs['j1Phi']]
                        libs_j1reco[b] = tmp_features[:,idxs['j1RecoLoss']]
                        libs_j1kl[b] = tmp_features[:,idxs['j1KlLoss']]
                    
                    else:
                        libs_j2pt[b] = np.concatenate((libs_j2pt[b],(tmp_features[:,idxs['j2Pt']])),axis=0) # WARNING: NP.LOG IF NOT NORMALIZED
                        libs_j2eta[b] = np.concatenate((libs_j2eta[b],tmp_features[:,idxs['j2Eta']]),axis=0)
                        libs_j2phi[b] = np.concatenate((libs_j2phi[b],tmp_features[:,idxs['j2Phi']]),axis=0)
                        libs_j2reco[b] = np.concatenate((libs_j2reco[b],tmp_features[:,idxs['j2RecoLoss']]),axis=0)
                        libs_j2kl[b] = np.concatenate((libs_j2kl[b],tmp_features[:,idxs['j2KlLoss']]),axis=0)
                        
                        libs_j1pt[b] = np.concatenate((libs_j1pt[b],(tmp_features[:,idxs['j1Pt']])),axis=0) # WARNING: NP.LOG IF NOT NORMALIZED
                        libs_j1eta[b] = np.concatenate((libs_j1eta[b],tmp_features[:,idxs['j1Eta']]),axis=0)
                        libs_j1phi[b] = np.concatenate((libs_j1phi[b],tmp_features[:,idxs['j1Phi']]),axis=0)
                        libs_j1reco[b] = np.concatenate((libs_j1reco[b],tmp_features[:,idxs['j1RecoLoss']]),axis=0)
                        libs_j1kl[b] = np.concatenate((libs_j1kl[b],tmp_features[:,idxs['j1KlLoss']]),axis=0)



            final_arr = None

            for b,lower in enumerate(binning):    
                print('Bin',b,'contains',len(evts_mjj[b]))

            
            for b,lower in enumerate(binning):
                #if b > 1:
                #    break
                
                # Now set up the KD Tree
                data_for_kdtree = np.stack((libs_j2pt[b],libs_j2eta[b]),axis=-1)
                data_in_bin = np.stack((evts_j2pt[b],evts_j2eta[b]),axis=-1)
                kdtrees[b] = KDTree(data_for_kdtree)        
                
                data_for_kdtree1 = np.stack((libs_j1pt[b],libs_j1eta[b]),axis=-1)
                data_in_bin1 = np.stack((evts_j1pt[b],evts_j1eta[b]),axis=-1)
                kdtrees1[b] = KDTree(data_for_kdtree1)        
                            
                # Expand dimensions
                evts_mjj[b] = np.expand_dims(evts_mjj[b],axis=1)
                evts_deta[b] = np.expand_dims(evts_deta[b],axis=1)
                evts_j1pt[b] = np.expand_dims(np.exp(evts_j1pt[b]),axis=1)
                evts_j1eta[b] = np.expand_dims(evts_j1eta[b],axis=1)
                evts_j1reco[b] = np.expand_dims(evts_j1reco[b],axis=1)
                evts_j1kl[b] = np.expand_dims(evts_j1kl[b],axis=1)
                evts_j2pt[b] = np.expand_dims(np.exp(evts_j2pt[b]),axis=1)
                evts_j2eta[b] = np.expand_dims(evts_j2eta[b],axis=1)
                evts_j2reco[b] = np.expand_dims(evts_j2reco[b],axis=1)
                evts_j2kl[b] = np.expand_dims(evts_j2kl[b],axis=1)
                
                print(f"Bin {b+1}/{len(binning)}")
                
                j1Total=evts_j1reco[b]+0.5*evts_j1kl[b]
                j2Total=evts_j2reco[b]+0.5*evts_j2kl[b]
                evt_disc=np.minimum(j1Total,j2Total)
                k_NN=1

                # Perform at least once
                print(f"Starting with k={k_NN}")
                dist,ind = kdtrees[b].query(data_in_bin, k=k_NN)
                flat_ind = ind.flatten()

                dist1,ind1 = kdtrees1[b].query(data_in_bin1, k=k_NN)
                flat_ind1 = ind1.flatten()

                replaced_j2reco = np.expand_dims(libs_j2reco[b][ind].mean(axis=1),axis=1)
                replaced_j2kl = np.expand_dims(libs_j2kl[b][ind].mean(axis=1),axis=1)
                
                replaced_j1reco = np.expand_dims(libs_j1reco[b][ind1].mean(axis=1),axis=1)
                replaced_j1kl = np.expand_dims(libs_j1kl[b][ind1].mean(axis=1),axis=1)
                
                replaced_j1Total=replaced_j1reco+0.5*replaced_j1kl
                replaced_j2Total=replaced_j2reco+0.5*replaced_j2kl
                replaced_evt_disc=np.minimum(replaced_j1Total,replaced_j2Total)
                    
                print(f"k={k_NN} | Before and after: {evt_disc.mean():4f} and {replaced_evt_disc.mean():4f}")

                best=[replaced_j1reco,replaced_j1kl,replaced_j1Total,replaced_j2reco,replaced_j2kl,replaced_j2Total,k_NN]
                    
                while (True):
                    if (k_NN==1 and evt_disc.mean()>replaced_evt_disc.mean()): break
                    
                    k_NN=k_NN*2
                    if (k_NN>128): break
                    
                    print(f"Increasing nearest neighbours to {k_NN}")
                    
                    dist,ind = kdtrees[b].query(data_in_bin, k=k_NN)
                    flat_ind = ind.flatten()

                    dist1,ind1 = kdtrees1[b].query(data_in_bin1, k=k_NN)
                    flat_ind1 = ind1.flatten()

                    replaced_j2reco = np.expand_dims(libs_j2reco[b][ind].mean(axis=1),axis=1)
                    replaced_j2kl = np.expand_dims(libs_j2kl[b][ind].mean(axis=1),axis=1)
                    
                    replaced_j1reco = np.expand_dims(libs_j1reco[b][ind1].mean(axis=1),axis=1)
                    replaced_j1kl = np.expand_dims(libs_j1kl[b][ind1].mean(axis=1),axis=1)
                    
                    replaced_j1Total=replaced_j1reco+0.5*replaced_j1kl
                    replaced_j2Total=replaced_j2reco+0.5*replaced_j2kl
                    
                    replaced_evt_disc=np.minimum(replaced_j1Total,replaced_j2Total)
                    print(f"k={k_NN} | Before and after: {evt_disc.mean():4f} and {replaced_evt_disc.mean():4f}")

                    
                    if (min(replaced_j1Total.mean(),replaced_j2Total.mean())<min(best[2].mean(),best[5].mean())):
                        print(f"Change: f{min(replaced_j1Total.mean(),replaced_j2Total.mean())} and best: {min(best[2].mean(),best[5].mean())}")
                        best=[replaced_j1reco,replaced_j1kl,replaced_j1Total,replaced_j2reco,replaced_j2kl,replaced_j2Total,k_NN]
                    pdb.set_trace()
                    if (min(replaced_j1Total.mean(),replaced_j2Total.mean())<min(j1Total.mean(),j2Total.mean())):
                        break # Break if loss is lowered
                
                print(f"Choosing k={best[6]} as the best case")
                replaced_j1reco=best[0];replaced_j1kl=best[1];replaced_j1Total=best[2]
                replaced_j2reco=best[3];replaced_j2kl=best[4];replaced_j1Total=best[5]

                print("##############################")
                
                
                #pdb.set_trace()
                #print(replaced_j2reco.shape)
                #print(replaced_j2kl.shape)
                #print(evts_j2pt[b].shape)
                
                #final_features_in_bin = np.squeeze(np.stack((evts_mjj[b],evts_deta[b],evts_j1pt[b],evts_j1eta[b],replaced_j1reco,replaced_j1kl,\
                #                                            evts_j2pt[b],evts_j2eta[b],replaced_j2reco,replaced_j2kl),axis=-1),axis=1)
                
                ### STORING SOME ADDITIONAL FEATURES SUCH AS TOTAL LOSS PER JET
                final_features_in_bin = np.squeeze(np.stack((evts_mjj[b],evts_deta[b],evts_j1pt[b],evts_j1eta[b],\
                                                            evts_j1reco[b]+0.5*evts_j1kl[b],replaced_j1reco,replaced_j1kl,replaced_j1reco+0.5*replaced_j1kl,\
                                                                evts_j2pt[b],evts_j2eta[b],evts_j2reco[b]+0.5*evts_j2kl[b],\
                                                                replaced_j2reco,replaced_j2kl,\
                                                                replaced_j2reco+0.5*replaced_j2kl),axis=-1),axis=1)
                #print(final_features_in_bin)
                #print(final_features_in_bin.shape)
                
                cols = ['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta','j1TotalLossBefore','j1RecoLoss', 'j1KlLoss','j1TotalLossReplaced',
                'j2Pt', 'j2Eta','j2TotalLossBefore','j2RecoLoss', 'j2KlLoss','j2TotalLossReplaced']
            
                ### NOTE THAT RecoLoss and KlLoss are the replaced values after mixing, not the pre-mixing values. 
                #print(final_features_in_bin)
                #print(final_features_in_bin.shape)

                if b == 0:
                    final_arr = final_features_in_bin
                else:
                    final_arr = np.concatenate((final_arr,final_features_in_bin),axis=0)

            #print(final_arr.shape)
            #out_filename=paths.sample_file_path(mixed_id,additional_id=sig_sample_id,mkdir=True,overwrite=True,customname=f'data_MIXED_{sig_sample_id}_{inj}_normalized_dynamicNN_v2')
            out_filename = f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}/qcd_sig_orig_RECO_injected_jets/sig_MCOrig_QR_Reco_MIXED_dynamicNN.h5'
            print("Complete. Now creating dataset")
            print(f"Output file: {out_filename}")
            
            ofile = h5py.File(out_filename, 'w')
            ofile.create_dataset('eventFeatures', data=final_arr)
            dt = h5py.special_dtype(vlen=str)
            #feature_names=np.array(['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta', 'j1RecoLoss', 'j1KlLoss',\
            #                                                'j2Pt', 'j2Eta', 'j2RecoLoss', 'j2KlLoss'],dtype=dt)
            
            feature_names=np.array(cols,dtype=dt)
            
            ofile.create_dataset('eventFeatureNames', data=feature_names)
            ofile.close()


        # Now end loop over values of injected signal int. lumi
    # End loop over injected sample IDs

    print("FAREWELL")    
    return None

if __name__ == '__main__':
    # parser = argparse.ArgumentParser("")
    # parser.add_argument('--eventlibrary', type=str, default="X", help="")
    # options = parser.parse_args()

    binning = [1450, 1529, 1604, 1681, 1761, 1844, 1930, 2019, 
               2111, 2206, 2305, 2406, 2512, 2620, 2733, 2849, 2969, 3093,
               3221, 3353, 3490, 3632, 3778, 3928, 4084, 4245, 4411, 4583,
               4760, 4943, 5132, 5327, 5527] ## 1450 and 1475 have been added
    
    #binning = [1450, 2450, 3450, 4450, 5450]

    create_libraries(binning)
