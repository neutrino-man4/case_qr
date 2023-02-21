#!/bin/bash

for inj in 0.0 #0.1 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
do
	echo "--------------------------------------------------------------"
	echo "doing injection ${inj}"
	python3 main_cms_vkfold_lmfit_scan.py -i ${inj} 2>&1 | tee logs/log_${inj}.txt
	echo "--------------------------------------------------------------"
	echo "done injection ${inj}"
	echo "--------------------------------------------------------------"
done	
