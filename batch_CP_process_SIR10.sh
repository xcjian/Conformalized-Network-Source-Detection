#!/bin/bash
## src = 10
#load data
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPSIR10.log 2>&1 &
wait
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow9.log 2>&1 &
wait