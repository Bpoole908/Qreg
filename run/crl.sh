########################################################################################
# Run Config
########################################################################################
# Enable from false to true to run
if false ; then
output_dir=3k-test
task=minihack_room
experiment=general/300k/qreg-nw-live2k-updates2k.yaml

### Task args ###
continual_testing_freq=null # Not reported in tag name
num_timesteps=3000 # Not reported in tag name

HYDRA_FULL_ERROR=1 python main.py -m \
+experiment=${experiment} \
exp=${task} \
++exp.exp_loader.continual_testing_freq=${continual_testing_freq} \
++exp.num_timesteps=${num_timesteps}
fi

# Enable from false to true to run
if false ; then
output_dir=3k-test
task=flappy
experiment=general/300k/qreg-nw-live2k-updates2k.yaml

### Task args ###
continual_testing_freq=null # Not reported in tag name
num_timesteps=3000 # Not reported in tag name

HYDRA_FULL_ERROR=1 python main.py -m \
+experiment=${experiment} \
exp=${task} \
++exp.exp_loader.exp_kwargs.continual_testing_freq=${continual_testing_freq} \
++exp.num_timesteps=${num_timesteps}
fi