"""
MIT License

Copyright (c) 2020 SamNPowers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@inproceedings{cora2022,
    title={CORA: Benchmarks, Baselines, and Metrics as a Platform for Continual Reinforcement Learning Agents},
    author={Powers*, Sam and Xing*, Eliot and Kolve, Eric and Mottaghi, Roozbeh and Gupta, Abhinav},
    booktitle={Conference on Lifelong Learning Agents (CoLLAs)},
    year={2022},
}
"""

import sys
import os
from time import time
from datetime import timedelta
from pdb import set_trace

import hydra
import torch
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.utils import Utils

from crl.common.yaml import HydraDict, set_omega_resolvers

def get_policy_and_experiment_from_config(config, output_dir):
    """
    The config dictionary will tell us which policy to use, which will allow us to populate the correct config file.
    """
    # Extract the spec of the experiment we will be running
    experiment = config['task_seq'].pop('exp_loader')
    if callable(experiment):
        experiment = experiment() 

    # Extract the configuration of the policy we will be running
    policy_config = config.pop("policy")
    policy_struct = policy_config.pop("policy_struct")
    policy_kwargs = policy_config.pop("policy_kwargs")
    
    # Get policy config and check validity of policy kwargs
    policy_config = policy_struct.config().load_from_dict(policy_kwargs)

    # Set output directory for continual RL. Hydra takes care of folder creation
    # thus the 'output_dir' just needs to be popped.
    experiment.set_output_dir(output_dir)
    policy_config.set_output_dir(output_dir)

    # Create policy 
    policy = policy_struct.policy(policy_config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)
        
    return experiment, policy

@hydra.main(version_base=None, config_path='configs/continual_rl', config_name='template')
def main(config):
    start_time = time()
    hydra_cfg = HydraConfig.get()
    print(f"Current working directory: {os.getcwd()}")
    print(f"Original working directory: {get_original_cwd()}")
    
    # Pytorch multiprocessing requires either forkserver or spawn.
    try:
        multiprocessing.set_start_method("spawn")
    except ValueError as e:
        # Windows doesn't support forking, so fall back to spawn instead
        assert "cannot find context" in str(e)
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        # Prevent "RuntimeError: context has already been set" when using hydra --multirun
        pass

    # Instantiate OmegaConf and store in HydraDict
    exp_config = HydraDict(config)
    
    # Build continual_rl experiment and policy
    experiment, policy = get_policy_and_experiment_from_config(config=exp_config, output_dir=os.getcwd())

    logger = Utils.create_logger(f"{policy.config.output_dir}/main.log")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Original working directory: {get_original_cwd()}")
    logger.info(f"Start time: {start_time}")
    
    # TODO: Temp fix to prevent continual rl experiment specs from having string task ids
    #       replay buffer can not store string task ids without throwing errors. Also
    #       algorithms like Packnet assume task ids are integers.
    for i, task in enumerate(experiment.tasks):
        if isinstance(task.task_id, str):
            logger.info(f"Task id changed from {task.task_id} to {i}")
            task.task_id = i
            task._task_spec._task_id = i
            experiment.tasks[i]._continual_eval_task_spec._task_id = i
            experiment.task_ids[i] = i

    # Only test to see if config builds properly 
    if config.validate_config:
        return

    if experiment is None:
        raise RuntimeError("No experiment started. Most likely there is no new run to start.")
    try:
        summary_writer = SummaryWriter(log_dir=experiment.output_dir)
        experiment.try_run(policy, summary_writer=summary_writer)
    finally:
        end_time = time()
        total_time = end_time-start_time
        logger.info(f"End time: {end_time}")
        logger.info(f"Elapsed time: {str(timedelta(seconds=total_time))}")
        # Clean stop any processes stuck due to an error
        sys.exit()

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    set_omega_resolvers()
    main()
   