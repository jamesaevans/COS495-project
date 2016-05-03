# salloc srun --pty $SHELL -l

# for non default change the code below to your liking
salloc --gres=gpu:1 --mem=16G srun --pty $SHELL -l
