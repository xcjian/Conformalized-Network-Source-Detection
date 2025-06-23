#!/bin/bash

nframe=${1:-16} # number of frames to sample. default: 16
ls=${2:-10}   # number of sequences, len_seq. default: 4000
gtype=highSchool  # graph type
stype=${3:-SIR}  # simulation type, SIR, SEIR, etc. default: SIR
R0=${4:-2.5}     # R0
beta=${5:-0.3}   # beta
gamma=${6:-0.4}  # gamma. If <=0, then the SI model is deployed.
f0=0.02    # min outbreak fraction
T=30      # simulation uplimit
n_sources=${7:-[[1, 1.0],[7, -1]]}
skip=${8:-[[1,2],[7,1]]} # set the skip step for different nsrc.

python sim_T.py --T ${T} --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --beta ${beta} --gamma ${gamma} --f0 ${f0} --n_sources ${n_sources} --skip ${skip}
