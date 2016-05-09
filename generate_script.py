import glob

text = r'''#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ajsun@princeton.edu
#SBATCH -o REPLACE_output.txt

source venv/bin/activate
python statefarm_train.py REPLACE.txt validate.txt'''

for file in glob.glob('p*.txt'):
    rep = file[0:4]
    out = open(rep + '.sh', 'w')
    new_script = text.replace("REPLACE", rep)
    out.write(new_script)
    out.close()