#!/bin/bash
# Mismatch of nsrc
# R0: [1, 15]
# True nsrc: [5, 10]  
## Estimated nsrc: [10, 15]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est10-15 --prop_model SIR > loadCPhighSchoolovernsrc.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est10-15 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchoolovernsrcpow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est10-15 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchoolovernsrcpow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est10-15 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchoolovernsrcpow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est10-15 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchoolovernsrcpow9.log 2>&1 &
wait

## Estimated nsrc: [1, 5]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-5 --prop_model SIR > loadCPhighSchoolundernsrc.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-5 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchoolundernsrcpow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-5 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchoolundernsrcpow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-5 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchoolundernsrcpow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-5 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchoolundernsrcpow9.log 2>&1 &
wait

## Estimated nsrc: [1, 15]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-15 --prop_model SIR > loadCPhighSchoolinclnsrc.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchoolinclnsrcpow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchoolinclnsrcpow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchoolinclnsrcpow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16 --save_exp_name nsrc5-10est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchoolinclnsrcpow9.log 2>&1 &
wait

# Mismatch of R0
# nsrc: [1, 15]
# True R0: [11, 25]  

## Estimated R0: [21, 35]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est21-35 --prop_model SIR > loadCPhighSchooloverR0.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est21-35 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchooloverR0pow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est21-35 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchooloverR0pow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est21-35 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchooloverR0pow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est21-35 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchooloverR0pow9.log 2>&1 &
wait

## Estimated R0: [1, 15]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-15 --prop_model SIR > loadCPhighSchoolunderR0.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchoolunderR0pow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchoolunderR0pow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchoolunderR0pow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-15 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchoolunderR0pow9.log 2>&1 &
wait

## Estimated R0: [1, 35]

### load data:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-35 --prop_model SIR > loadCPhighSchoolinclR0.log 2>&1 || true
wait

### Conformal prediction:
python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-35 --set_recall 1 --set_prec 1 --pow_expected 0.3 --prop_model SIR > logfiles/runCPhighSchoolinclR0pow3.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-35 --set_recall 1 --set_prec 1 --pow_expected 0.5 --prop_model SIR > logfiles/runCPhighSchoolinclR0pow5.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-35 --set_recall 1 --set_prec 1 --pow_expected 0.7 --prop_model SIR > logfiles/runCPhighSchoolinclR0pow7.log 2>&1 &

python -u main_mismatch.py --graph highSchool --train_exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls1000_nf16 --save_exp_name R011-25est1-35 --set_recall 1 --set_prec 1 --pow_expected 0.9 --prop_model SIR > logfiles/runCPhighSchoolinclR0pow9.log 2>&1 &
wait