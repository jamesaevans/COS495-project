salloc srun --pty $SHELL -l

# for non default change the code below to your liking
#salloc --gres=gpu:1 -c 2 --mem=4G srun --pty $SHELL -l