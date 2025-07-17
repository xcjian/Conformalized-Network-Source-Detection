#!/bin/bash

# SIR over highSchool
## src = 1
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 1.0 --prop_model SIR > loadCPSIR1.log 2>&1 &
wait
## src = 7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPSIR7.log 2>&1 &
wait
## src = 14
python -u main.py --graph highSchool --train_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPSIR14.log 2>&1 &
wait

# randomSIR over highSchool
## Slow
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPhighSchoolSlow.log 2>&1 &
wait
## Mid
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPhighSchoolMid.log 2>&1 &
wait
## Fast
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPhighSchoolFast.log 2>&1 &
wait

# randomSIR over bkFratB
## Slow
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPbkFratBSlow.log 2>&1 &
wait
## Mid
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPbkFratBMid.log 2>&1 &
wait
## Fast
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPbkFratBFast.log 2>&1 &
wait

# randomSIR over sfhh
## Slow
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPsfhhSlowpow3.log 2>&1 &
wait
## Mid
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPsfhhMidpow3.log 2>&1 &
wait
## Fast
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR > loadCPsfhhFastpow3.log 2>&1 &
wait