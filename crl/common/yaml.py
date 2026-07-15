import os
import warnings
from os.path import join
from pdb import set_trace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, get_method

class HydraDict(dict):
    """A dict populated by Hydra-instantiating a raw config subtree.

    Wraps `hydra.utils.instantiate` so a nested Hydra config node (containing
    `_target_`-style entries) can be used directly as a plain dict once its
    values have been instantiated.
    """

    def __init__(self, raw, convert='all'):
        """Instantiates `raw` via Hydra and populates this dict with the result.

        Args:
            raw: The raw (un-instantiated) Hydra config node to instantiate.
            convert: The `_convert_` mode passed to `hydra.utils.instantiate`
                (e.g. 'all' to convert OmegaConf containers to native types).
        """
        self._raw = raw
        init = instantiate(raw, _convert_=convert)
        super().__init__(**init)


class GetRunID():
    """OmegaConf resolver for Hydra that determines the current experiment's run ID.

    Registered under the `get_run_id` resolver name (see `set_omega_resolvers`).
    Run IDs are consecutive integers assigned per save path; this resolver either
    recovers the ID from Hydra's working directory (if Hydra already created a
    numbered run directory) or computes the next free ID by scanning existing
    numbered subdirectories of the save path.
    """

    def __call__(self, path, ids_to_exist=[]):
        """Determines the run ID to use for `path`.

        Args:
            path: The save directory whose numbered subdirectories are used to
                infer prior run IDs.
            ids_to_exist: Additional run IDs (e.g. of runs about to be created)
                to treat as already taken when computing the next free ID.

        Returns:
            The run ID as an int: either the current Hydra working directory's
            numeric name (if Hydra has already changed into a numbered run
            directory under `path`), or the next free run ID under `path`.
        """
        cwd = os.getcwd()
        cwd_basename = os.path.basename(cwd)
        
        # Account for Hydra changing directory by checking if
        # rel path is in current working directory
        if path in cwd and cwd_basename.isdigit():
            return int(cwd_basename)
        
        run_id = self._get_proper_run_id(path, ids_to_exist)
        return int(run_id)

    def _get_proper_run_id(self, save_path, ids_to_exist):
        """Computes the next free run ID for `save_path`.

        Args:
            save_path: Directory to scan for existing numbered run subdirectories.
            ids_to_exist: Additional run IDs to treat as already taken.

        Returns:
            The smallest run ID (int) not already used by an existing
            subdirectory or by `ids_to_exist`; 0 if none exist yet.
        """
        save_path = Path(save_path)
        prior_run_ids = []
        
        # Only check for prior ids if path exists
        if save_path.exists():
            prior_run_ids = self._find_prior_run_ids(save_path.iterdir())
        else:
            msg = f"Can not find prior run ids as save path {str(save_path)!r} does not exist."
            warnings.warn(msg)
        
        # Combine prior ids and future ids to get all ids that will exist
        all_ids = np.unique(np.hstack([prior_run_ids, ids_to_exist]))
        # If no ids will exist the first id should be 0 then
        if len(all_ids) == 0:
            return 0
        # Find all the potential next IDs
        potential_next_ids = self._get_next_run_id(all_ids)
        # If no potential ids are found then +1 to the last ID
        # otherwise take the first potential ID (smallest next ID)
        if len(potential_next_ids) == 0:
            return all_ids[-1] + 1
        else:
            return potential_next_ids[0]
        
    def _find_prior_run_ids(self, folders):
        """Extracts run IDs from directory entries whose names are integers.

        Args:
            folders: Iterable of `Path` entries (e.g. from `Path.iterdir()`).

        Returns:
            A sorted NumPy array of the run IDs found among `folders`.
        """
        prior_run_ids = np.array([int(f.name) for f in folders if str(f.name).isdigit()])
        return np.sort(prior_run_ids)

    def _get_next_run_id(self, all_ids):
        """Finds run IDs below the current max that are not yet used.

        Args:
            all_ids: Array of run IDs already taken (prior runs plus reserved IDs).

        Returns:
            A sorted array of unused run IDs in `[0, max(all_ids)]`; empty if all
            are taken (in which case the caller should use `max(all_ids) + 1`).
        """
        potential_ids = np.arange(0, all_ids[-1]+1)
        return np.setdiff1d(potential_ids, all_ids)


def convert_float_to_str(num):
    """Formats a number as a short human-readable string with k/m suffixes.

    Args:
        num: A number (or numeric string) to format, e.g. for use in experiment
            tags. May be None.

    Returns:
        None if `num` is None; otherwise a string like '1.5m' (millions),
        '2.3k' (thousands), or the plain number as a string if below 1000.
    """
    if num is None:
        return None
    
    if isinstance(num, str):
        num = float(num)

    if num >= 1_000_000:
        return f"{np.format_float_positional(round(num / 1_000_000, 2), trim='-')}m"
    elif num >= 1_000:
        return f"{np.format_float_positional(round(num / 1_000, 2), trim='-')}k"
    else:
        return str(num)


def call_module(module_path, *args):
    """Looks up a callable by its dotted module path and calls it with `args`.

    Args:
        module_path: Dotted path to a function/method, resolvable by
            `hydra.utils.get_method`.
        *args: Positional arguments to pass to the resolved callable.

    Returns:
        The return value of calling the resolved callable with `args`.
    """
    module = get_method(module_path)
    return module(*args)


def set_omega_resolvers(
    use_eval=True,
    use_join=True,
    use_get_run_id=True,
    use_getcwd=True,
):
    """Registers this project's custom OmegaConf resolvers for use in Hydra configs.

    Always registers 'get_none', 'get_class', 'float_to_str', and 'call_module';
    the resolvers below are individually toggleable since they have side effects
    or environment dependencies:

    Args:
        use_eval: If True, registers 'eval' (evaluates a Python expression string).
        use_join: If True, registers 'join' (joins path components via `os.path.join`).
        use_get_run_id: If True, registers 'get_run_id' (a `GetRunID` instance).
        use_getcwd: If True, registers 'getcwd' (returns `os.getcwd()`).
    """
    if use_eval:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    if use_join:
        OmegaConf.register_new_resolver("join", lambda *args: join(*args), replace=True)
    if use_get_run_id:
        OmegaConf.register_new_resolver("get_run_id", GetRunID(), replace=True)
    if use_getcwd:
        OmegaConf.register_new_resolver("getcwd", os.getcwd, replace=True)
    OmegaConf.register_new_resolver("get_none", lambda: None, replace=True)
    OmegaConf.register_new_resolver("get_class", get_class, replace=True)
    OmegaConf.register_new_resolver("float_to_str", convert_float_to_str, replace=True)
    OmegaConf.register_new_resolver("call_module", call_module, replace=True)