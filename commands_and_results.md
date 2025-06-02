# SI model

## 1 source:

* Command:

`nohup python -u main_SIR_T.py --gt highSchool --prop_model SI --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 50 --ks 2 --kt 1 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --start 2 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.9434 --val_pct 0.0189 > runSI1.log 2>&1 &`

* Results:

valid f1: 0.216

test f1: 0.227 

test prec: 0.247 

test recall: 0.225

## 7 sources:

* Command:

`nohup python -u main_SIR_T.py --gt highSchool --prop_model SI --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 50 --ks 2 --kt 1 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.9434 --val_pct 0.0189 > run_SI7.log 2>&1 &`

* Results:

validation f1: 0.345

test f1, prec, recall:

0.331 0.241 0.529

## 10 sources:

`nohup python -u main_SIR_T.py --gt highSchool --prop_model SI --n_node 774 --n_frame 16 --batch_size 16 --epoch 50 --ks 2 --kt 1 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16 --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5.0 --train_pct 0.9434 --val_pct 0.0189 > run_SI10.log 2>&1 &`

* Results:

validation f1: 0.352

test f1, prec, recall:

0.343 0.254 0.532

# SIR model:

## 1 source:

`nohup python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 50 --ks 6 --kt 2 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --start 2 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.9434 --val_pct 0.0189 > runSIR1.log 2>&1 &`

20 epochs are enough.

note: although I run 50 epochs in the beginning, the model was not updated after the 20-th epoch.

* Results:

validation f1: 0.333

test f1: 0.318

test prec: 0.323

test recall: 0.321

## 7 sources:

`nohup python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > runSIR_7.log 2>&1 &`

* Results:

validation f1: 0.417

test f1uracy: 0.399

test precision: 0.362

test recall: 0.451

## 14 sources:

`nohup python -u main_SIR_T.py --gt highSchool --prop_model SIR --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 6 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_entire.pickle --exp_name SIR_nsrc14_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --start 1 --end -1 --random 0 --lr 0.001 --valid 1 --pos_weight 5 --train_pct 0.9434 --val_pct 0.0189 > run_SIR14.log 2>&1 &`

* Results:

validation f1: 0.408

test f1uracy: 0.396

test precision: 0.418

test recall: 0.379