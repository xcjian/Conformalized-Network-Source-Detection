#!/bin/bash

nframe=${1:-16} # number of frames to sample
ls=${2:-21200}   # number of sequences, len_seq
gtype=highSchool  # graph type
stype=${3:-SIR}  # simulation type, SIR, SEIR, etc
nsrc_lo=${4:-1} # range of number of sources
nsrc_hi=${5:-15}
R0_lo=${6:-5} # range of R0
R0_hi=${7:-35} 
gamma_lo=${8:-0.1} # range of R0
gamma_hi=${9:-0.4} 
skip=${10:-1} # first available instance
f0=0.02   # min outbreak fraction

python -u sim_entire_mix_random.py --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --f0 ${f0}\
 --nsrc_lo ${nsrc_lo}\
 --nsrc_hi ${nsrc_hi}\
 --R0_lo ${R0_lo}\
 --R0_hi ${R0_hi}\
 --gamma_lo ${gamma_lo}\
 --gamma_hi ${gamma_hi}\
 --skip ${skip}\