#!/bin/bash
# export every artifact chart's data (run in miles.sif on a vis node)
F=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/figures; O=$F/export/json; mkdir -p $O
for s in fcs/parity fcs/probe fcs/easyppo_s42 fcs/screen fcs/blind_eval19 eval_curve trust_region_dynamics sc/sc_vs_main q38/overview q38/heldout_all; do
  python3 $F/export/export_fig.py $F/$s.py $O/$(basename $s).json 2>&1 | grep -E "exported|Error|Traceback|line [0-9]+" | tail -3
done
