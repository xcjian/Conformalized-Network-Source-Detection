#!/bin/bash
# SI over highSchool
## src = 7
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI7pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI7pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI7pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI7pow9.log 2>&1 &

## src = 10
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI10pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI10pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI10pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SI --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSI10pow9.log 2>&1 &
wait

# SIR over highSchool
## src = 1
#pow = 1
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 1.0 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR1pow1.log 2>&1 &
## src = 7
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR7pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR7pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR7pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR7pow9.log 2>&1 &
## src = 14
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR14pow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR14pow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR14pow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPSIR14pow9.log 2>&1 &
wait

# randomSIR over highSchool
## Slow
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolSlowpow9.log 2>&1 &
## Mid
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolMidpow9.log 2>&1 &
wait
## Fast
#pow = 0.3
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPhighSchoolFastpow9.log 2>&1 &

# randomSIR over bkFratB
## Slow
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBSlowpow9.log 2>&1 &
wait
## Mid
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBMidpow9.log 2>&1 &
## Fast
#pow = 0.3
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph bkFratB --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPbkFratBFastpow9.log 2>&1 &
wait
# randomSIR over sfhh
## Slow
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhSlowpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhSlowpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhSlowpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhSlowpow9.log 2>&1 &
## Mid
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhMidpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhMidpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhMidpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhMidpow9.log 2>&1 &
wait
## Fast
#pow = 0.3
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.3 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhFastpow3.log 2>&1 &
#pow = 0.5
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhFastpow5.log 2>&1 &
#pow = 0.7
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.7 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhFastpow7.log 2>&1 &
#pow = 0.9
python -u main.py --graph sfhh --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls21200_nf16 --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --pow_expected 0.9 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > runCPsfhhFastpow9.log 2>&1 &
wait