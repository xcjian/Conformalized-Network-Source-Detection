#!/bin/bash

nframe=${1:-16} # number of frames to sample. default: 16
ls=${2:-4000}   # number of sequences, len_seq. default: 4000
gtype=highSchool  # graph type
stype=SIR  # simulation type, SIR, SEIR, etc. default: SIR
R0=${3:-2.5}     # R0
beta=${4:-0.3}   # beta
gamma=${5:-0.4}  # gamma. If <=0, then the SI model is deployed.
f0=0.02    # min outbreak fraction
T=30      # simulation uplimit
n_sources=${6:-3}

python sim_T.py --T ${T} --n_frame ${nframe} --len_seq ${ls} --graph_type ${gtype} --sim_type ${stype} --R0 ${R0} --beta ${beta} --gamma ${gamma} --f0 ${f0} --n_sources ${n_sources}
