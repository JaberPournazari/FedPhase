#!/bin/bash
# Final step: collect every group + dataset_overview.py + paper_results.py -> results/data1/paper/
source "$(dirname "$0")/lib.sh"
P=$(sbatch --parsable $(_dep "$DEP") job_paper.slurm); echo "paper -> $P (log: hpc/<group>/logs/paper_$P.out)"
