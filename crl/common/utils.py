"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import cloudpickle
import functools
import random
from typing import Dict, Tuple, Union
from pdb import set_trace

import torch
import numpy as np
from gym import spaces


def set_torch_random_seed(seed: int = None, cuda_deterministic: bool = False) -> int:
    """Seeds Python, NumPy, and Torch RNGs, generating a random seed if none is given.

    Args:
        seed: Seed to use for the random generators. If None, a random seed is
            generated from `os.urandom`.
        cuda_deterministic: If True and a GPU is being used, forces CuDNN to use
            deterministic algorithms (disabling its benchmark mode), which can hurt
            performance but makes results reproducible.

    Returns:
        The seed that was used (either the one passed in, or the generated one).
    """
    seed = int.from_bytes(os.urandom(4), byteorder="big") if seed is None else seed
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if cuda_deterministic:
        # Deterministic operations for CuDNN, it may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
  
    return seed


def debug_tensor_image(image, filename, ext='jpg'):
    """Saves a single image tensor to disk as an image file, for debugging.

    Args:
        image: A tensor convertible to a PIL image via `torchvision.transforms.ToPILImage`.
        filename: Output path (without extension).
        ext: File extension/format to save as (e.g. 'jpg', 'png').
    """
    import torchvision
    import torchvision.transforms as T
    import matplotlib.pyplot as plt

    transform = T.ToPILImage()
    img = transform(image)
    img.save(f'{filename}.{ext}')


def pickle_dump(data, filepath):
    """Serializes `data` to `filepath` using cloudpickle.

    Args:
        data: Any cloudpickle-serializable object.
        filepath: Destination file path.
    """
    with open(filepath, 'wb') as fp:
        cloudpickle.dump(data, fp)


def pickle_load(filepath):
    """Loads and returns a cloudpickle-serialized object from `filepath`.

    Args:
        filepath: Path to a file previously written by `pickle_dump`.

    Returns:
        The deserialized object.
    """
    with open(filepath, 'rb') as fp:
        data = cloudpickle.load(fp)
    return data


def rgetattr(obj, path: str, delim='.', *default):
    """Recursively resolves a dotted attribute path on an object.

    Args:
        obj: The root object to resolve attributes from.
        path: Dotted attribute path, e.g. 'attr1.attr2.etc'.
        delim: Delimiter used to split `path` into individual attribute names.
        *default: Optional single default value to return if any attribute in
            the path is missing, instead of raising.

    Returns:
        The value of `obj.attr1.attr2...` as resolved by walking `path`.

    Raises:
        AttributeError: If an attribute in the path is missing and no default
            was provided.
    """
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def reduce_dims(data, ordered_pairs, allow_1D=False):
    """Reshapes `data` by collapsing groups of its dimensions together.

    Args:
        data: Torch tensor or NumPy array whose dimensions will be reshaped.
        ordered_pairs: A list of ints/lists where each entry names one or more
            existing dimension indices to merge (via product of their sizes) into
            a single new dimension. The order of `ordered_pairs` determines the
            order of the resulting dimensions. Any dimensions of `data` not
            mentioned are appended unchanged at the end.
        allow_1D: If False and the resulting shape would be 1D, an extra trailing
            dimension of size 1 is added to keep the result 2D. If True, a 1D
            result is left as-is.

    Returns:
        `data` reshaped according to `ordered_pairs`.

    Raises:
        ValueError: If `ordered_pairs` references a dimension more than once, or
            references a dimension that does not exist.
    """
    new_dims = []
    shape = np.array(data.shape)
    shape_idx = {i for i in range(len(shape))}
    used_shape_idx = set()

    for pair in ordered_pairs:
        # Track which indexes have been used and check they are valid
        pair_set = set([pair]) if isinstance(pair, int) else set(pair) 
        inter = pair_set.intersection(shape_idx)
        if inter.intersection(used_shape_idx):
            raise ValueError(f"order_pairs {ordered_pairs} contains duplicate dims.")
        if len(inter) == 0:
            raise ValueError(f"Dimensions {pair} do not exists.")
        used_shape_idx |= inter
        
        # Compute new dimension size
        new_dim_size = shape[pair] if isinstance(pair, int) else shape[pair].prod()
        new_dims.append(new_dim_size)
    
    # Check if all indices were given, if not add them to the end
    diff = shape_idx.difference(used_shape_idx)
    if diff != 0:
        new_dims.extend(shape[list(diff)])
    
    # Keep shape 2D
    if not allow_1D and len(new_dims) == 1:
        new_dims.append(1)
        
    return data.reshape(*new_dims)


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """Gets the shape of an observation space (useful for sizing replay buffers).

    Args:
        observation_space: A gym `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`,
            or `Dict` space.

    Returns:
        A shape tuple, or (for `Dict` spaces) a dict mapping each sub-space's key
        to its shape tuple.

    Raises:
        NotImplementedError: If `observation_space` is not one of the supported
            gym space types.
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    
    
def get_action_dim(action_space: spaces.Space) -> int:
    """Gets the flattened dimensionality of an action space.

    Args:
        action_space: A gym `Box`, `Discrete`, or `MultiBinary` space (with 1D `n`).

    Returns:
        The number of scalar values needed to represent one action.

    Raises:
        NotImplementedError: If `action_space` is not one of the supported gym
            space types.
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    
    
def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """Resolves a requested device string to an available PyTorch device.

    Falls back to CPU if CUDA is requested (or 'auto', which defaults to CUDA)
    but not available.

    Args:
        device: One of 'auto', 'cuda', 'cpu', or an existing `torch.device`.
            'auto' is treated as a request for 'cuda'.

    Returns:
        A `torch.device` that is actually available on this machine.
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_cycle_regions(start, end, cycle_size, num_cycles):
    """Computes the [start, end] step range of a task's region in each of several cycles.

    Args:
        start: Start step of the task's region within the first cycle.
        end: End step of the task's region within the first cycle.
        cycle_size: Number of steps per full cycle (used as the offset between
            cycles).
        num_cycles: Number of cycles to generate regions for.

    Returns:
        A list of `[start, end]` step pairs, one per cycle, each offset by
        `i * cycle_size`.
    """
    return [[start+(i*cycle_size), end+(i*cycle_size)] for i in range(num_cycles)]
