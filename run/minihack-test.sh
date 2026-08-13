########################################################################################
# Minihack Sequences
########################################################################################
output_dir=3k-test # Custom name for output directory within exps/
task=minihack_room # Options: minihack_room
experiment=qreg-nw-live2k-updates2k.yaml # Options: See configs under configs/continual_rl/experiment

### Task Arg Overrides ###
continual_testing_freq=null
num_timesteps=3000 

# Hydra command
HYDRA_FULL_ERROR=1 python main.py -m \
+experiment=${experiment} \
task_seq=${task} \
++task_seq.exp_loader.continual_testing_freq=${continual_testing_freq} \
++task_seq.num_timesteps=${num_timesteps}