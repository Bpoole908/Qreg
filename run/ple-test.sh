########################################################################################
# PLE Sequences
########################################################################################
output_dir=3k-test # Custom name for output directory within exps/
task=flappy # Options: flappy, catcher, flappy-single, catcher-single
experiment=qreg-nw-live2k-updates2k.yaml # Options: See configs under configs/continual_rl/experiment

### Task Arg Overrides ###
continual_testing_freq=null
num_timesteps=3000

# Hydra command
HYDRA_FULL_ERROR=1 python main.py -m \
+experiment=${experiment} \
task_seq=${task} \
++task_seq.exp_loader.exp_kwargs.continual_testing_freq=${continual_testing_freq} \
++task_seq.num_timesteps=${num_timesteps}