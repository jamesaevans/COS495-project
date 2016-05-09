#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ajsun@princeton.edu
#SBATCH -o p052_output.txt

source venv/bin/activate
python statefarm_train.py p052.txt validate.txt