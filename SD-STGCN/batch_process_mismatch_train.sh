#!/bin/bash
# nsrc mismatch
## over estimated
python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 3 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16_entire.pickle --exp_name SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.49505 --val_pct 0.009901 > Runoverestnrc.log 2>&1 &
wait
## under estimated
python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 3 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16_entire.pickle --exp_name SIR_nsrc1-5_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.49505 --val_pct 0.009901 > Rununderestnrc.log 2>&1 &
wait
## include
python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 5 --kt 4 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16_entire.pickle --exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.49505 --val_pct 0.009901 > Runestnrc.log 2>&1 &


# R0 mismatch
## over estimated
python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 5 --kt 2 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16_entire.pickle --exp_name SIR_nsrc1-15_Rzero21-35_gamma0.1-0.4_ls40400_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.49505 --val_pct 0.009901 > RunoverestR0.log 2>&1 &
wait
## under estimated
# not needed.
## include
python -u main_SIR_T.py --gt highSchool --pos_weight 5 --n_node 774 --n_frame 16 --batch_size 16 --epoch 20 --ks 5 --kt 3 --sconv gcn --save 1 --graph ./dataset/highSchool/data/graph/highSchool.edgelist --pred ./output/models/highSchool/pred_highSchool_nf16.pickle --seq ./dataset/highSchool/data/SIR/SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16_entire.pickle --exp_name SIR_nsrc1-15_Rzero1-35_gamma0.1-0.4_ls40400_nf16 --start 1 --end -1 --random 0 --lr 1e-3 --valid 1 --train_pct 0.49505 --val_pct 0.009901 > RunincludeR0.log 2>&1 &
wait