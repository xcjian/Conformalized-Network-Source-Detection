#!/bin/bash
# SI over highSchool
## src = 7
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI7pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI7pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI7pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI7pow9.log 2>&1 &

## src = 10
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI10pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI10pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI10pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSI10pow9.log 2>&1 &
wait

# SIR over highSchool
## src = 1
#pow = 1
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 1.0 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR1pow1.log 2>&1 &
## src = 7
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR7pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR7pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR7pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR7pow9.log 2>&1 &
## src = 10
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow9.log 2>&1 &
wait

# randomSIR over highSchool
## Slow
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolSlowpow9.log 2>&1 &
## Mid
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolMidpow9.log 2>&1 &
wait
## Fast
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPhighSchoolFastpow9.log 2>&1 &

# randomSIR over bkFratB
## Slow
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBSlowpow9.log 2>&1 &
wait
## Mid
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBMidpow9.log 2>&1 &
## Fast
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPbkFratBFastpow9.log 2>&1 &
wait
# randomSIR over sfhh
## Slow
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhSlowpow9.log 2>&1 &
## Mid
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhMidpow9.log 2>&1 &
wait
## Fast
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPsfhhFastpow9.log 2>&1 &
wait