case_qr : quantile regression for anomalous physics
=================================================================================

For a vector quantile regression (all quantiles regressed simultaneously), the executable is `main_vkfold_lmfit_scan.py`. The lmfit library is used to smoothen the results after the QR. It also supports signal injection:

.. code-block:: bash

    python3 main_vkfold_lmfit_scan.py $signal_name $resonance_mass $inj_in_picobarn

e.g.,
    
.. code-block:: bash

    python3 main_vkfold_lmfit_scan.py grav_2p5_narrow 2500 0.09


