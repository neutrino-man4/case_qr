<!-- case_qr : quantile regression for anomalous physics
=================================================================================

For a vector quantile regression (all quantiles regressed simultaneously), the executable is ``main_cms_vkfold_lmfit_scan.py``. The lmfit library is used to smoothen the results after the QR. It also supports signal injection:

.. code-block:: bash

    python3 main_cms_vkfold_lmfit_scan.py $signal_name $resonance_mass $inj_in_picobarn

e.g.,
    
.. code-block:: bash

    python3 main_cms_vkfold_lmfit_scan.py grav_2p5_narrow 2500 0.09 -->

# CASE QR: Quantile Regression for Anomaly Detection

## Step 1: Signal Injection
To perform the signal injection, run the executable as follows. 
```python3 main_signal_injection.py ```

- The QCD signal region events are read from `base_dir/sample_dir['qcdSigMCOrigReco']`  defined in `case_paths/sample_dict_file_parts_reco.py`.

- Two additional keys have been added to the `sample_dir` dictionary in this updated version of `case_paths/sample_dict_file_parts_reco.py`
    - `sample_dir[qcdSigMCOrigReco_injectedJets]`: This specifies where the injected jets will be stored. A new folder will be created for each signal (whose path relative to `base_dir` and `sample dir` can be found from the values corresponding to the signal keys). Therefore, for different injected signal `WkkToWRadionToWWW_M3000_Mr170Reco` luminosity values, the final output `.h5` files can be found in the directory:
    ```base_dir/sample_dir['qcdSigMCOrigReco_injectedJets']/sample_dir['WkkToWRadionToWWW_M3000_Mr170Reco']/ ```
    - `sample_dir[qcdSigMCOrigReco_mixedJets]`: This is where the mixed jets will be written to (see Step 2 below). Just as for the injected jets, the output will be written to:
    ``` base_dir/sample_dir['qcdSigMCOrigReco_mixedJets']/sample_dir['WkkToWRadionToWWW_M3000_Mr170Reco']/```


## Step 2: Jet Mixing/Replacement
To perform the jet mixing, run the executable as follows:
``` python3 main_mix.py ```

Output is written to the directory mentioned above. Don't forget to place the additional keys to `case_paths/sample_dict_file_parts_reco.py` in case you have an older version of `case_paths`. 


## Step 3: Vector Quantile Regression
To perform the k-fold vector quantile regression (all quantiles regressed simultaneously) on the un-mixed jet samples, run as follows:
``` source runner.sh ```
This script calls `main_cms_vkfold_lmfit_scan.py` with different values of the injection parameter in $pb^{-1}$, which decides how many signal events are to be injected into the QCD SR dataset.
TODO: Generalize to all signals. 
