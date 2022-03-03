#!/bin/sh

# This scipt runs the deepcover on ImmunoBert for a given batch
# To be used by a parallel runner.

allele="HLA-A33-01"
exp_name="exp6"

echo Batch $1

python ../deepcover/src/deepcover.py --model "../ImmunoBERT/output/main/CONTEXT-PSEUDO-HEAD_Cls-DECOY_19-LR_0.00001/checkpoints/epoch=4-step=3648186.ckpt" --inputs ../deepcover/data/immunobert/alleles/$allele.npy --outputs "outs/$exp_name/$allele" --testgen-size 2000 --path_to_immunobert ../ImmunoBERT --batch_size 40 --batch_number $1