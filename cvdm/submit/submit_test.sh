#!/bin/bash
#SBATCH --job-name=cvdm             # job name
#SBATCH --nodes=1                   # request the number of nodes
#SBATCH --ntasks=1                  # number of tasks per node
#SBATCH --cpus-per-task=8           # number of cpus per task
#SBATCH --gres=gpu:1               # request the number of gpus
#SBATCH --time=20:00:00             # time limit to finish the job. Will be cancelled if not finished in this time
#SBATCH --partition=gpuq            # node partition

module load apptainer
module load cuda12.3/toolkit

apptainer exec --nv \
  --bind /cm/shared/apps/cuda12.3/toolkit/12.3.2:/cm/shared/apps/cuda12.3/toolkit/12.3.2 \
  --bind /projects/data_solutions/raiders/pathai:/mnt \
  /homes/seitzcx/git/cvdm/cvdm.sif \
  bash -c "
    export CUDA_HOME=/cm/shared/apps/cuda12.3/toolkit/12.3.2
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
    cd ~/git/cvdm/cvdm/mains
    source ~/git/cvdm/venv/bin/activate
    python test.py --config-path ~/git/cvdm/cvdm/configs/test/test_nanoruler.yaml
  "
