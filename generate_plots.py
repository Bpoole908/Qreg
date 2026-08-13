import os
from pdb import set_trace

import numpy as np
import hydra
from hydra.utils import get_original_cwd, instantiate
from hydra.core.hydra_config import HydraConfig

from crl.common.metrics import Metrics
from crl.common.yaml import HydraDict, set_omega_resolvers

@hydra.main(version_base=None, config_path='configs/plot', config_name='template')
def main(config):
    hydra_cfg = HydraConfig.get()
    print(f"Current working directory: {os.getcwd()}")
    print(f"Original working directory: {get_original_cwd()}")

    init_config = instantiate(config, _convert_='all')
    metrics = Metrics(init_config)
    metrics.visualize()

if __name__ == '__main__':
    set_omega_resolvers()
    main()
