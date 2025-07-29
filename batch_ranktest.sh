#!/bin/bash
# test over different (alpha, beta) configs
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.1 0.3 0.5 0.7 --alpha_ran 0.05 0.10 0.15
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.1 0.3 0.5 0.7 --alpha_ran 0.05 0.10 0.15
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.1 0.3 0.5 0.7 --alpha_ran 0.05 0.10 0.15
wait

# test over different number of sources
#SI 1 7 10
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --beta_ran 0.0 --alpha_ran 0.1
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --beta_ran 0.1 0.3 0.5 --alpha_ran 0.1
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16 --beta_ran 0.1 0.3 0.5 --alpha_ran 0.1
wait
#SIR 1 7 10
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --beta_ran 0.0 --alpha_ran 0.1
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --beta_ran 0.1 0.3 0.5 --alpha_ran 0.1
wait
python -u sign_test.py --graph highSchool --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --beta_ran 0.1 0.3 0.5 --alpha_ran 0.1
wait

# test over different graphs
python -u sign_test.py --graph bkFratB --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait
python -u sign_test.py --graph bkFratB --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait
python -u sign_test.py --graph bkFratB --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait
python -u sign_test.py --graph sfhh --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait
python -u sign_test.py --graph sfhh --test_exp_name SIR_nsrc1-15_Rzero11-25_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait
python -u sign_test.py --graph sfhh --test_exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls8000_nf16 --beta_ran 0.3 --alpha_ran 0.10
wait