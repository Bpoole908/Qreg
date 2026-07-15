**Code is currently not runnable as it is missing the environment conda file which contains forked repositories that would violate the double-blind review process. However, the core code for the algorithms is included.**

# Code Structure

## `crl/` package

| Path | Contents |
|---|---|
| `crl/common/utils.py` | Seeding, tensor/env helpers (`set_torch_random_seed`, `get_obs_shape`, `get_action_dim`, `get_device`, `reduce_dims`, `rgetattr`, pickle helpers) |
| `crl/common/yaml.py` | Custom OmegaConf/Hydra resolvers (`HydraDict`, `GetRunID`, `set_omega_resolvers`) used across `configs/` |
| `crl/common/metrics.py` | `Metrics`, `TabulateMetrics`, `TabulateTransferMetrics` — loads TensorBoard event data, computes CORA-style forgetting/transfer metrics, renders plots and LaTeX/Excel tables |
| `crl/common/plotting.py` | `get_cycle_regions()` — computes shaded training-region bounds for plots |
| `crl/common/buffers/replay_buffers.py` | `Samples`, `BaseBuffer`, `ReplayBuffer` — the standard fixed-size replay buffer (adapted from Stable-Baselines3) |
| `crl/common/buffers/rehearsal_replay_buffer.py` | `RehearsalReplayBuffer`, `RRBManager` — a curated buffer of past samples with cached model outputs, used by the data-rehearsal policies |
| `crl/common/envs/gym_ple.py` | `GymPLE` — Gym wrapper around a PyGame Learning Environment (PLE) game |
| `crl/common/envs/ple_envs.py` | `ContinualCatcher`, `ContinualFlappyBird` — parameterized PLE games used to build task families |
| `crl/experiment_loader.py` | `exp_loader()` — builds a `continual_rl` `Experiment` from a list of games/tasks |
| `crl/experiments/make_ple.py` | `make_ple()`, `wrap_ple()`, `get_single_ple_task()` — builds a single Atari-style-preprocessed PLE task |
| `crl/policies/` | One subpackage per continual-learning method (see below) |

### `crl/policies/` — one subpackage per method

Each subpackage pairs a `*Policy` (subclasses `continual_rl`'s `PolicyBase`) with a `*PolicyConfig`
(subclasses `continual_rl`'s `ConfigBase`) and a loss module. Most subclass the base DQN policy
rather than reimplementing training from scratch.

| Subpackage | Policy / Config | Method |
|---|---|---|
| `dqn/` | `DQNPolicy` / `DQNPolicyConfig` | Base DQN (Mnih et al., 2015) with a `dqn`/`double_dqn` loss switch (`DQNLosses`) — the base class for every other DQN-derived policy below |
| `ewc/` | `EWCDQNPolicy` / `EWCDQNPolicyConfig` | DQN + Online Elastic Weight Consolidation (regularizes towards an online Fisher-weighted snapshot of prior weights) |
| `l2/` | `L2DQNPolicy` / `L2DQNPolicyConfig` | DQN + L2 weight regularization (encoder only) against the prior task's weights |
| `mer/` | `MERDQNPolicy` / `MERDQNPolicyConfig` | DQN + Meta-Experience Replay (nested Reptile meta-updates) |
| `packnet/` | `PackNetDQNPolicy` / `PackNetDQNPolicyConfig` | DQN + PackNet (prunes and retrains after each task to reserve capacity for the next) |
| `data_rehearsal/` | `DRDQNPolicy` / `DRDQNPolicyConfig` (DQN-based) | This project's "qreg" method: maintains a `RehearsalReplayBuffer` of cached embeddings/Q-values and regularizes the model towards them. |

### Root scripts

| Script | Purpose |
|---|---|
| `main.py` | Hydra entrypoint (`configs/continual_rl/`). Builds a policy + `continual_rl` experiment from the resolved config and runs training. |

## Configuration Structure

All three entrypoints use [Hydra](https://hydra.cc/), each with its own independent config tree under
`configs/`. Config groups are selected on the command line (`group=name`), and Hydra composes the selected
YAML files together (later files override earlier ones) into the final config object passed to the
`@hydra.main`-decorated function.

### `configs/continual_rl/` — used by `main.py`

- **`template.yaml`** — the root config. Declares `policy` and `exp` as *required* groups (`defaults:` lists
  them as `???`, so a run must supply both). Also sets up output-directory templating
  (`job_name`, `run_dir`, `run_id`) and disables Hydra's own logging/directory side effects.
- **`policy/*.yaml`** — one file per method (`dqn.yaml`, `ddqn.yaml`, `ewc.yaml`, `l2.yaml`,
  `mer.yaml`, `packnet.yaml`, `data-rehearsal.yaml`). Each has two blocks:
  - `policy_kwargs`: the hyperparameters passed into `<Method>PolicyConfig.load_from_dict()`. **Every key
    must exactly match an attribute name set in that config class's `__init__`** (see
    `crl/policies/*/*_policy_config.py`) — this mapping is by exact name only (via
    `ConfigBase._auto_load_class_parameters` in the `continual_rl` submodule), so a misspelled or renamed
    key doesn't fail silently: it's left over after parsing and raises `UnknownExperimentConfigEntry`.
  - `policy_struct`: points (via `hydra.utils.get_class`) at the concrete `*PolicyConfig` and `*Policy`
    Python classes to instantiate.
- **`exp/*.yaml`** — one file per continual-learning task/environment setup (`catcher.yaml`, `flappy.yaml`,
  and`minihack_room.yaml`). Each defines an
  `exp_loader` (pointing at `crl.experiment_loader.exp_loader`) plus:
  - `task_func` / `task_func_kwargs` — the function that builds one task (e.g.
    `crl.experiments.make_ple.get_single_ple_task`) and its per-task keyword overrides — e.g. a different
    `fall_speed_modifier` per task, turning one game into a family of related-but-distinct tasks.
  - `game_names` — the environment class to use per task (e.g. `crl.common.envs.ple_envs.ContinualCatcher`).
  - `exp_kwargs` — passed straight to `continual_rl`'s `Experiment` (e.g. `cycle_count`).
- **`experiment/**/*.yaml`** — "recipe" configs that bundle a `policy` override with tuned hyperparameters,
  invoked via `+experiment=<group>/<name>` on the command line (e.g. `+experiment=general/300k/qreg.yaml`).
  Organized by benchmark (`general/300k/`), with further subfolders like `search/` for
  hyperparameter-sweep variants.

Putting it together, a run is specified by supplying `policy` and `exp` directly:

```
python main.py policy=dqn exp=catcher job_name=my-first-run
```

or by using a bundled recipe (which already overrides `policy` and tunes hyperparameters) plus a task:

```
python main.py +experiment=general/300k/qreg.yaml exp=catcher
```

**Naming and run IDs.** Every `*_policy_config.py` module also defines an `experiment_tag(policy, exp)`
function that builds a short, human-readable tag from the run's *non-default* hyperparameters (e.g.
`rb=50k_lr=0.0001`). `template.yaml`'s `name` field calls this via the custom `call_module` OmegaConf
resolver, so output directories self-document their configuration. Repeated runs of the same `job_name`
are kept separate by an auto-incrementing `run_id`, computed by the `get_run_id` resolver
(`crl.common.yaml.GetRunID`), which scans the target directory for the next free integer.

**Output layout:** `exps/<output_dir>/<job_name>/<run_id>/`.