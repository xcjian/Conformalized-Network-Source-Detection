#!/bin/bash


Rzero=${3:-2.5} # simulation R0
beta=${4:-0.1}   # beta
gamma=${5:-0.4} # simulation gamma
ns=${2:-4000}   # num of sequences
nf=${1:-16}     # num of frames
N=774    # num of nodes in graph

gt=highSchool     # random graph type, ER, BA, BA-Tree, RGG

bs=16     # batch_size
ep=3      # num of epochs
lr=1e-3   # learning rate
ks=4      # spatio kernel size
# kt=3      # temporal kernel size
kt=1      # temporal kernel size
sc=gcn    # spatio convolution layer type
save=1    # save every # of epochs

skip=${6:-1} # start from skip-th snapshot
end=-1 # the sampled snapshots will end at (skip + n_frame)-th snapshot

T=30

random=0 # randomly sample n_frame snapshots?

g="./dataset/${gt}/data/graph/${gt}.edgelist"
p="./output/models/${gt}/pred_${gt}_${gs}_nf${nf}.pickle"
s="./dataset/${gt}/data/SIR/SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}_entire.pickle"
exp_name="SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}"

python main_SIR_T.py --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --pred $p --seq $s --exp_name $exp_name --start ${skip} --end ${end} --random ${random}\
    --lr ${lr}

