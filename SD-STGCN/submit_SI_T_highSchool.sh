#!/bin/bash


Rzero=2.5 # simulation R0
gamma=0.4 # simulation gamma
ns=4000   # num of sequences
nf=16     # num of frames
N=774    # num of nodes in graph

gt=highSchool     # random graph type, ER, BA, BA-Tree, RGG

bs=16     # batch_size
ep=3      # num of epochs
ks=4      # spatio kernel size
kt=3      # temporal kernel size
sc=gcn    # spatio convolution layer type
save=1    # save every # of epochs

skip=1
end=-1

T=30

g="./dataset/${gt}/data/graph/${gt}.edgelist"
p="./output/models/${gt}/pred_${gt}_${gs}_nf${nf}.pickle"
s="./dataset/${gt}/data/SI/SI_Rzero${Rzero}_gamma${gamma}_T${T}_ls${ns}_nf${nf}_entire.pickle"

python main_SI_T.py --gt ${gt} --n_node ${N} --n_frame ${nf} --batch_size ${bs} --epoch ${ep} --ks ${ks} --kt ${kt} --sconv ${sc} --save ${save} --graph $g --pred $p --seq $s --start ${skip} --end ${end}

