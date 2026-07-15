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

import os
import collections
import copy
from typing import Dict, List
from pathlib import Path
from pdb import set_trace

import cloudpickle as pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import sem


class TabulateMetrics():
    """Builds a (eval task x training task/cycle) table of a single metric, plus grand averages.

    Given per-task metric data (as produced by `Metrics.compute_metrics`), lays
    it out into a 2D table indexed by evaluation task (rows) and by
    training-task/cycle (columns), with an extra final row/column holding the
    grand average (and its standard error) across the other entries.
    """

    def __init__(
        self,
        task_tags: List[str],
        num_cycles: int,
        metric_scale: int = 10
    ):
        """
        Args:
            task_tags: One tag string per task, used only to determine the
                number of tasks (`num_tasks`).
            num_cycles: Number of task cycles represented in the table columns.
            metric_scale: Multiplier applied to raw metric values (and their
                SEMs) before storing them in the table, e.g. to display as
                percentages.
        """
        self.task_tags = task_tags
        self.num_cycles = num_cycles
        self.metric_scale = metric_scale

        self.num_tasks = len(task_tags)

    def __call__(self, model_metrics, metric_key):
        """Builds `self.table`/`self.error_table` for one metric across all tasks.

        Args:
            model_metrics: Dict mapping task tag to a dict of metric name ->
                per-run data, as produced by `Metrics.compute_metrics`.
            metric_key: Which metric (e.g. 'forgetting', 'transfer') to tabulate.
        """
        self.init_tables()

        for task_id, tag in enumerate(self.task_tags):
            task_data = model_metrics[tag]
            self.aggregate(task_data[metric_key], task_id=task_id)

        self.consolidate()

    def init_tables(self):
        """Allocates empty `table`/`error_table` and resets the running aggregates."""
        # Pre-allocate our tables (inverted so shape is eval task by training tasks)
        self.table = [[None for _ in range(self.num_tasks * self.num_cycles + 1)] for _ in range(self.num_tasks + 1)]
        self.error_table = [[None for _ in range(self.num_tasks * self.num_cycles + 1)] for _ in range(self.num_tasks + 1)]

        # To ensure our standard error of the mean statistics are using independent data, we average the metrics over the run across the appropriate axis
        # Data for grand average over evaluation task across all training tasks and cycles (last row in table)
        self.task_id_run_aggregates = {}  
        # Data for grand average of how training on a task transfers to all other tasks
        self.eval_id_run_aggregates = {} 
        
    def aggregate(self, data, task_id):
        """Fills in one eval task's row of `table`/`error_table` and accumulates grand-average data.

        Args:
            data: Dict mapping training task id to a dict mapping cycle id to a
                list of per-run metric values for evaluation task `task_id`.
            task_id: The evaluation task these values belong to (row index).
        """
        # For each training task (+ cycle id)
        for train_task_id in range(self.num_tasks):
            impact_data = data.get(train_task_id, {})

            # For each cycle
            for cycle_id in range(self.num_cycles):
                impact_cycle_run_data = impact_data.get(cycle_id, None)
                impact_cycle_data = sum(impact_cycle_run_data) / len(impact_cycle_run_data) if impact_cycle_run_data is not None else None
                impact_cycle_error = sem(impact_cycle_run_data) if impact_cycle_run_data is not None else None
                # print("\t", impact_cycle_data, impact_cycle_error)
                self.table[task_id][cycle_id * self.num_tasks + train_task_id] = impact_cycle_data * self.metric_scale if impact_cycle_data is not None else None
                self.error_table[task_id][cycle_id * self.num_tasks + train_task_id] = impact_cycle_error * self.metric_scale if impact_cycle_error is not None else None
           
                # Save individual run statistics for grand averages. 
                if impact_cycle_run_data is not None:
                    for run_id in range(len(impact_cycle_run_data)):
                        if task_id not in self.task_id_run_aggregates:
                            self.task_id_run_aggregates[task_id] = {}

                        if run_id not in self.task_id_run_aggregates[task_id]:
                            self.task_id_run_aggregates[task_id][run_id] = []
                        # Computed using a single task id
                        self.task_id_run_aggregates[task_id][run_id].append(impact_cycle_run_data[run_id])

                        if train_task_id not in self.eval_id_run_aggregates:
                            self.eval_id_run_aggregates[train_task_id] = {}

                        if run_id not in self.eval_id_run_aggregates[train_task_id]:
                            self.eval_id_run_aggregates[train_task_id][run_id] = []
                        # Accumulates over different task id's
                        self.eval_id_run_aggregates[train_task_id][run_id].append(impact_cycle_run_data[run_id])

    def consolidate(self):
        """Computes grand-average row/column/corner values and converts tables to arrays.

        Fills in the final row (grand average per training task, across eval
        tasks) and final column (grand average per eval task, across training
        tasks/cycles) of `table`/`error_table`, plus the single bottom-right
        corner cell (average of all grand averages). Also converts both tables
        from nested lists to float NumPy arrays.
        """
        for task_id in range(self.num_tasks):
            # Grand average for evaluation task across all training tasks and cycles
            if task_id in self.task_id_run_aggregates:
                metric_values = np.array(list(self.task_id_run_aggregates[task_id].values())).mean(axis=1)
                self.table[task_id][-1] = self.metric_scale * metric_values.mean(axis=0)  # Along the axes for consistency with sem
                self.error_table[task_id][-1] = self.metric_scale * sem(metric_values)
            # Grand average of how training on a task transfers to all other tasks
            if task_id in self.eval_id_run_aggregates:
                metric_values = np.array(list(self.eval_id_run_aggregates[task_id].values())).mean(axis=1)
                self.table[-1][task_id] = self.metric_scale * metric_values.mean(axis=0)
                self.error_table[-1][task_id] = self.metric_scale * sem(metric_values)
                
        self.table = np.array(self.table, dtype=float)
        self.error_table = np.array(self.error_table, dtype=float)
        
        # Average of ALL grand averages (not very useful)
        all_task_id_run_agg = {}
        for task_id in self.task_id_run_aggregates.keys():
            for run_id in self.task_id_run_aggregates[task_id].keys():
                if run_id not in all_task_id_run_agg:
                    all_task_id_run_agg[run_id] = []
                    
                all_task_id_run_agg[run_id].extend(self.task_id_run_aggregates[task_id][run_id])
    
        all_task_id_run_agg = np.array(list(all_task_id_run_agg.values())).mean(axis=1)
        self.table[-1][-1] = self.metric_scale * all_task_id_run_agg.mean()
        self.error_table[-1][-1] = self.metric_scale * sem(all_task_id_run_agg)    
    
# TODO: Transfer only computes 1 cycle worth of metrics, can this be computed over multiple?
class TabulateTransferMetrics(TabulateMetrics):
    """`TabulateMetrics` specialized for forward transfer, tabulated as (eval task x training task).

    Unlike the base class, forward transfer only has one value per (eval task,
    training task) pair rather than per cycle, so the table is square
    (`num_tasks + 1` on each side) instead of having `num_cycles` columns per task.
    """

    def __init__(self, task_tags, num_cycles, metric_scale):
        """See `TabulateMetrics.__init__`."""
        super().__init__(task_tags, num_cycles, metric_scale)

    def __call__(self, model_metrics, metric_key='forward_transfer'):
        """See `TabulateMetrics.__call__`; defaults `metric_key` to 'forward_transfer'."""
        super().__call__(model_metrics, metric_key)

    def init_tables(self):
        """Allocates a square `table`/`error_table` and resets the running aggregates."""
        # Pre-allocate our tables
        self.table = [[None for _ in range(self.num_tasks + 1)] for _ in range(self.num_tasks + 1)]
        self.error_table = [[None for _ in range(self.num_tasks + 1)] for _ in range(self.num_tasks + 1)]

        # To ensure our standard error of the mean statistics are using independent data, we average the metrics over the run across the appropriate axis
        self.task_id_run_aggregates = {}  # For a given task id, aggregate the run id data (across eval)
        self.eval_id_run_aggregates = {}  # For a given eval id, aggregate the run id data (across task)
        
    def aggregate(self, data, task_id):
        """Fills in one training task's row of `table`/`error_table` and accumulates grand-average data.

        Args:
            data: Dict mapping eval task id to a list of per-run forward-transfer
                values attributable to training task `task_id`.
            task_id: The training task these values belong to (row index).
        """
        for eval_id in range(self.num_tasks):
            impact_cycle_run_data = data.get(eval_id, None)

            impact_data = sum(impact_cycle_run_data) / len(impact_cycle_run_data) if impact_cycle_run_data is not None else None
            impact_error = sem(impact_cycle_run_data) if impact_cycle_run_data is not None else None

            self.table[task_id][eval_id] = impact_data * self.metric_scale if impact_data is not None else None
            self.error_table[task_id][eval_id] = impact_error * self.metric_scale if impact_data is not None else None

            # Aggregate statistics holding the task_id and eval_id constant:
            # First we average the data over the same run, to give us a per-run forgetting statistic
            if impact_cycle_run_data is not None:
                for run_id in range(len(impact_cycle_run_data)):
                    # Aggregate by task_id
                    if task_id not in self.task_id_run_aggregates:
                        self.task_id_run_aggregates[task_id] = {}

                    if run_id not in self.task_id_run_aggregates[task_id]:
                        self.task_id_run_aggregates[task_id][run_id] = []

                    self.task_id_run_aggregates[task_id][run_id].append(impact_cycle_run_data[run_id])

                    # Aggregate by eval
                    if eval_id not in self.eval_id_run_aggregates:
                        self.eval_id_run_aggregates[eval_id] = {}

                    if run_id not in self.eval_id_run_aggregates[eval_id]:
                        self.eval_id_run_aggregates[eval_id][run_id] = []

                    self.eval_id_run_aggregates[eval_id][run_id].append(impact_cycle_run_data[run_id])


class Metrics(object):
    """Loads TensorBoard event data for an experiment, computes CORA-style CL metrics, and plots/exports them.

    Given a description of an experiment (which models were run, which tasks
    they were trained/evaluated on, and where the event files live), this class
    loads the raw reward curves (`visualize` -> `read_exp_data`), post-processes
    them (smoothing/clipping), computes forgetting/transfer/return metrics
    (`compute_metrics`), and produces both Plotly figures and LaTeX/Excel metric
    tables (`plot_models`, `plot_metrics`).
    """

    def __init__(self, experiment_data):
        """
        Args:
            experiment_data: A dict describing where to find event files and how
                to process them. Keys such as 'save_in_cache' and 'use_cache'
                are popped during construction; the remainder is kept as `self._exp_data` 
                and consumed by `visualize`/`compute_metrics`/`plot_models`. 
                It should contain:

                (generally common)
                    tag_base: event file tag (e.g. 'eval_reward').
                    legend_size: font size for the legend (e.g. 30).
                    title_size: font size for the title (e.g. 40).
                    axis_size: font size for the axes (e.g. 20).
                    axis_label_size: font size for the axis labels (e.g. 30).
                    plot_ext: Extension to be used when saving plots.

                (generally experiment-specific)
                    models: model_data, see below.
                    tasks: tasks_list, see below.
                    exp_dir: the directory where experiment data is stored.
                    rolling_mean_count: the rolling mean window size (e.g. 20).
                    filter: how to filter the data ('ma' for moving average, 'ema'
                        for one-sided exponential moving average, or 'smooth').
                    num_cycles: how many cycles to analyze (e.g. 5).
                    num_task_steps: number of steps per task (e.g. 5e6).
                    exp_name: friendly name for caching purposes (e.g. 'procgen'),
                        used to be called which_exp.
                    xaxis_tickvals: the ticks to use when displaying the x-axis
                        (e.g. list(np.arange(0, 150e6 + 1, 30e6))).

                where `model_data` maps model name to a dict of model info, e.g.:
                    {
                        "MODEL_NAME": {
                            runs: paths to the runs for the experiment
                                (e.g. [f'path_to_data_for_each_run/run_{i}/' for i in range(20)]),
                            color: color to display on plots (e.g. 'rgba(77, 102, 133, 1)'),
                            color_alpha: alpha for displaying on plots (e.g. 0.2),
                            line: Plotly line kwargs for customizing the line,
                        }
                    }
                and `tasks_list` maps task name to a dict of task info, e.g.:
                    TASKS_ROOM = [
                        {
                            name: 'Task 1'
                            i: event id corresponding to the task for training data (e.g. 0),
                            eval_i: event id corresponding to the task for eval data (e.g. 1),
                            y_range: the range of values for y (e.g. [0., 1.25]),
                            yaxis_dtick: the tick size for the y axis (e.g. 0.25),
                            train_regions: the regions for which this task was trained
                                (e.g. [[5e6 * i, 5e6 * (i + 1)] for i in range(0, 6 * 5, 6)]),
                        }
                    ]
        """
        self._exp_data = experiment_data
        self._save_in_cache = self._exp_data.pop('save_in_cache', True)
        self._use_cache = self._exp_data.pop('use_cache', False)
    
    def visualize(self, plot_spec=None):
        """Loads, processes, and plots reward curves and metrics for every configured model.

        For each model: reads its event data, post-processes it (smoothing/clipping),
        computes forgetting/transfer/return metrics, and combines runs together
        for plotting. Finally renders both the reward-curve plots (`plot_models`)
        and the metric tables (`plot_metrics`).

        Args:
            plot_spec: Optional dict of overrides merged into `self._exp_data`
                before processing (e.g. to tweak plot styling for one call).
        """
        if plot_spec is not None:
            self._exp_data.update(plot_spec)
    
        tags = []
        for task_v in self._exp_data['tasks']:
            tags.append(f"{self._exp_data['tag_base']}/{task_v['i']}")
            if 'eval_i' in task_v.keys():
                tags.append(f"{self._exp_data['tag_base']}/{task_v['eval_i']}")
            
        print(f'tags: {tags}')
    
        combined_data = {}
        raw_data= {}
        all_metrics = {}
        for model_k, model_v in self._exp_data['models'].items():
            print(f'loading data for model: {model_k}')
            data = self.read_exp_data(model_v, tags)
            data = self.post_processing(data, tags)

            # Compute the metrics after we've smoothed (so our values are more representative) but before we interpolate
            # to combine the runs together
            raw_data[model_k] = data
            all_metrics[model_k] = self.compute_metrics(data)
            combined_data[model_k] = self.combine_exp_data(data, tags)
 
        self.plot_models(combined_data)
        self.plot_metrics(all_metrics)
        
        
    def read_exp_data(self, model_v, tags):
        """Loads (or reads from cache) the raw event data for every run of one model.

        Args:
            model_v: The model info dict (see `Metrics.__init__`), used for its
                'runs' list of run directory names.
            tags: The list of event-file tags (train/eval reward keys) to extract.

        Returns:
            Dict mapping run id to that run's event data (dict mapping tag to a
            list of (step, value) pairs).

        Raises:
            FileNotFoundError: If a run's directory does not exist under `exp_dir`.
            RuntimeError: If no matching event files are found for a run.
        """
        all_run_data = {}

        for run_id in model_v['runs']:
            # check if cached data exists
            target_file = pattern = os.path.join(self._exp_data['exp_dir'], f'{run_id}') 
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"Path {target_file!r} does not exist.")
            metrics_cache_file = os.path.join(target_file, 'event-metrics.pkl')
            if self._use_cache and os.path.exists(metrics_cache_file):
                print(f'loading cached: {metrics_cache_file}')
                event_data = pickle.load(open(metrics_cache_file, 'rb'))
            else:
                import expt
                print(f'finding event files for {target_file}')
                # BUG: Any metric reported by an RLR algorithm will have duplicate steps (due to continual rl).
                #      expt does not keep all duplicate steps, only the last one.
                runs = expt.get_runs(target_file)
                event_dfs = pd.concat([r.df for r in runs])
                event_dfs = event_dfs.rename(columns={'global_step':'step'})
                event_dfs = event_dfs.melt(var_name='tag', id_vars='step')
                event_dfs = event_dfs.dropna().reset_index(drop=True).set_index('tag').loc[tags]
    
                if len(event_dfs) == 0:
                    raise RuntimeError(f'no event files found: {pattern}')

                print(f"collecting event data from event files...")
                event_data = self.collate_event_df(event_dfs)

                if self._save_in_cache:
                    print(f'caching event file to {metrics_cache_file}')
                    pickle.dump(event_data, open(metrics_cache_file, 'wb'))

            all_run_data[run_id] = event_data
        return all_run_data
    
    def post_processing(self, data, tags):
        """Clips and smooths raw event data for each run/tag.

        Applies the configured `clip_y_range` (if any) and one of the
        'ma' (moving average), 'ema' (one-sided exponential moving average), or
        'smooth' filters (per `self._exp_data['filter']`) to every run's series.

        Args:
            data: Dict mapping run id to dict mapping tag to a list of
                (step, value) pairs, as returned by `read_exp_data`.
            tags: The tags to process for each run; tags missing from a run are
                skipped.

        Returns:
            Dict with the same structure as `data`, with values smoothed/clipped.

        Raises:
            ValueError: If `self._exp_data['filter']` is not one of 'ma', 'ema',
                or 'smooth'.
        """
        post_processed_data = {}
        for run_id, d in data.items():
            new_d = {}
            for k in tags:
                if k not in d:
                    continue
    
                run = d[k]
    
                xs = np.array([run_datum[0] for run_datum in run])
                ys = [run_datum[1] for run_datum in run]
    
                if self._exp_data.get("clip_y_range", None) is not None:
                    clip_range = self._exp_data["clip_y_range"]
                    ys = np.array(ys).clip(min=clip_range[0], max=clip_range[1])
                if self._exp_data['filter'] == 'ma':
                    rolling_accumulator = collections.deque(maxlen=self._exp_data['rolling_mean_count'])
                    for x_id, x in enumerate(xs):
                        rolling_accumulator.append(ys[x_id])
                        ys[x_id] = np.array(rolling_accumulator).mean()
                elif self._exp_data['filter'] == 'ema':
                    xs, ys, _ = self.one_sided_ema(np.array(xs), np.array(ys), n=50)
                elif self._exp_data['filter'] == 'smooth':
                    ys = self.smooth(ys, self._exp_data['rolling_mean_count'], mode='causal')
                else:
                    raise ValueError
                processed_run = list(zip(xs, ys))
    
                new_d[k] = processed_run
    
            post_processed_data[run_id] = new_d
        return post_processed_data
    
    def one_sided_ema(self, xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
        """Performs one-sided (causal) EMA smoothing, resampled onto an even grid.

        Does not extrapolate, so it assumes `xolds[0] <= low` and `high <= xolds[-1]`.

        Args:
            xolds: Array/list of x values, sorted in ascending order.
            yolds: Array/list of y values; must be the same length as `xolds`.
            low: Minimum value of the new x grid. Defaults to `xolds[0]`.
            high: Maximum value of the new x grid. Defaults to `xolds[-1]`.
            n: Number of points in the new x grid.
            decay_steps: EMA decay factor, expressed in new-x-grid steps.
            low_counts_threshold: y values whose EMA count falls below this
                threshold are set to NaN (insufficient nearby data).

        Returns:
            Tuple `(xnews, ys, count_ys)` where `xnews` is the new x grid, `ys` is
            the EMA of y at each point of `xnews`, and `count_ys` is the EMA of
            the y counts at each point of `xnews`.
        """
        low = xolds[0] if low is None else low
        high = xolds[-1] if high is None else high
    
        assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
        assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
        assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))
    
        xolds = xolds.astype('float64')
        yolds = yolds.astype('float64')
    
        luoi = 0 # last unused old index
        sum_y = 0.
        count_y = 0.
        xnews = np.linspace(low, high, n)
        decay_period = (high - low) / (n - 1) * decay_steps
        interstep_decay = np.exp(- 1. / decay_steps)
        sum_ys = np.zeros_like(xnews)
        count_ys = np.zeros_like(xnews)
        for i in range(n):
            xnew = xnews[i]
            sum_y *= interstep_decay
            count_y *= interstep_decay
            while True:
                if luoi >= len(xolds):
                    break
                xold = xolds[luoi]
                if xold <= xnew:
                    decay = np.exp(- (xnew - xold) / decay_period)
                    sum_y += decay * yolds[luoi]
                    count_y += decay
                    luoi += 1
                else:
                    break
            sum_ys[i] = sum_y
            count_ys[i] = count_y
    
        ys = sum_ys / count_ys
        ys[count_ys < low_counts_threshold] = np.nan
    
        return xnews, ys, count_ys
    
    def smooth(self, y, radius, mode='two_sided', valid_only=False):
        """Smooths signal `y` with a moving-average window of the given radius.

        Args:
            y: 1D array/list of values to smooth.
            radius: Half-width (in samples) of the averaging window.
            mode: 'two_sided' averages over
                `[max(index - radius, 0), min(index + radius, len(y)-1)]`; 'causal'
                averages over `[max(index - radius, 0), index]` only.
            valid_only: If True, sets entries where the full-sized window is not
                available to NaN instead of using a truncated window.

        Returns:
            The smoothed array, same length as `y`.
        """
        assert mode in ('two_sided', 'causal')
        if len(y) < 2*radius+1:
            return np.ones_like(y) * y.mean()
        elif mode == 'two_sided':
            convkernel = np.ones(2 * radius+1)
            out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
            if valid_only:
                out[:radius] = out[-radius:] = np.nan
        elif mode == 'causal':
            convkernel = np.ones(radius)
            out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
            out = out[:-radius+1]
            if valid_only:
                out[:radius] = np.nan
        return out
    
    def collate_event_df(self, event_df):
        """Groups a long-format event dataframe into per-tag (step, value) lists.

        Args:
            event_df: A dataframe with 'tag' and 'step' columns (plus a value
                column), as produced by melting the raw `expt` run dataframes.

        Returns:
            Dict mapping tag name to a list of `[step, value]` rows, sorted by step.
        """
        tag_values = {}
        # event_df = pd.concat(event_df_list)
        # Combine list using corresponding keys
        for name, grp_df in event_df.groupby('tag'):
            grp_df.sort_values('step', inplace=True)
            tag_values[name] = grp_df.values.tolist()

        return tag_values
    
    def compute_metrics(self, data):
        """Computes CORA-inspired continual-learning metrics for every task.

        For each task, computes forgetting (final/average/worst), transfer
        (final/average/worst), forward transfer, and average return, each
        scaled by the largest absolute return observed for that task.

        Args:
            data: Dict mapping run id to dict mapping tag to a list of
                (step, value) pairs, as returned by `post_processing`.

        Returns:
            Dict mapping task tag to a dict of metric name -> metric data (as
            returned by the corresponding `compute_*` method).
        """
        # Grab the tag ids we will use to evaluate the metrics: if we collected explicit eval data, use that.
        tags = self.get_metric_tags()
        num_tasks = len(tags)
        metrics = {}
    
        # For each task (labeled by a tag), grab all of the associated runs, then compute the metrics on them
        # tags contains task_id and tag which can be f'eval_reward/{task-id}' or f'train_reward/{task-id}'
        # data contains all data for a given experiment including multiple runs if passed.
        # run_data contains data for a given run of the experiment
        # pre_task_data contains the a given evaluation (current task_tag) across all runs 
        for task_id, task_tag in enumerate(tags):
            per_task_data = []
            for run_data in data.values():
                per_task_data.append(run_data[task_tag])
            
            # Scale by the largest (absolute) return seen for this task across all runs
            max_return = np.abs(np.concatenate([np.array([run[1] for run in task]) for task in per_task_data])).max()

            metrics[task_tag] = {
                "forgetting": self.compute_forgetting_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='final'
                ),
                "average_forgetting": self.compute_forgetting_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='average'
                ),
                "worst_forgetting": self.compute_forgetting_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='worst'
                ), 
                "forward_transfer": self.compute_forward_transfer_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    list(range(len(tags)))[:task_id],
                    return_scale=1/max_return
                ),
                'transfer': self.compute_transfer_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='final'
                ),
                'average_transfer': self.compute_transfer_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='average'
                ),
                'worst_transfer': self.compute_transfer_metric(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    task_id, 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1/max_return,
                    forgetting_type='worst'
                ),
                'average_return': self.compute_average_return(
                    per_task_data, 
                    self._exp_data["num_task_steps"], 
                    num_tasks,
                    num_cycles=self._exp_data.get("num_cycles_for_forgetting", 1), 
                    return_scale=1,
                ),
            }

        return metrics
    
    def get_metric_tags(self):
        """Gets the event tags used for metric computation, in task order.

        It is assumed that the order is consistent: i.e. tags A, B, C, D will be
        used to compute how much forgetting D causes for B and C.

        Returns:
            List of tags, one per task, in task order (using each task's
            'eval_i' id if present, else its 'i' id).
        """
        task_ids = [task["eval_i"] if "eval_i" in task else task["i"] for task in self._exp_data["tasks"]]
        tags = [f"{self._exp_data['tag_base']}/{id}" for id in task_ids]
        return tags
    
    def compute_forgetting_metric(
        self, 
        task_results, 
        task_steps, 
        task_id, 
        num_tasks, 
        num_cycles, 
        return_scale,
        forgetting_type='final'
    ) -> Dict[str, Dict[str, List]]:
        """Computes how much reward on `task_id` is lost while training on each subsequent task/cycle.

        For each subsequent task and cycle, compares the reward on `task_id`
        right before that task starts training to the reward on `task_id`
        (final/average/worst, per `forgetting_type`) during that task's training
        region.

        Args:
            task_results: Per-run list of (step, value) reward data for `task_id`.
            task_steps: Number of steps allotted per task.
            task_id: The evaluation task whose forgetting is being measured.
            num_tasks: Total number of tasks.
            num_cycles: Number of task cycles to consider.
            return_scale: Scale factor applied to raw reward values.
            forgetting_type: 'final', 'average', or 'worst', which reward
                within the subsequent task's training region to compare against.

        Returns:
            Dict mapping subsequent task id to a dict mapping cycle id to a list
            of per-run forgetting values for `task_id`.

            Example (for task_id = 0):
                Task 0:
                    Cycle 1: amount of forgetting for task ID 0
                Task 1:
                    Cycle 0: amount of forgetting for task ID 0
                    Cycle 1: amount of forgetting for task ID 0
                Task 2:
                    Cycle 0: amount of forgetting for task ID 0
                    Cycle 1: amount of forgetting for task ID 0

        Raises:
            ValueError: If `forgetting_type` is not 'final', 'average', or 'worst'.
        """
        per_run_forgetting_per_subsequent = {id: {} for id in range(num_tasks)}  # Inner dict maps cycle to total
    
        # Loop over runs for experiment
        for run_id, task_result in enumerate(task_results):
            xs = np.array([t[0] for t in task_result])
            ys = np.array([t[1] for t in task_result]) * return_scale

            # Loop over cycles  
            for cycle_id in range(num_cycles):
                for subsequent_task_id in range(num_tasks):
                    # It's not really "catastrophic forgetting" if we haven't seen the task yet, so skip the early tasks
                    if cycle_id == 0 and subsequent_task_id <= task_id:
                        continue
    
                    offset = cycle_id * num_tasks
                    
                    # Before training region for subsequent_task_id
                    before_task_rewards = self.get_rewards_for_region(xs, ys, [None, (subsequent_task_id + offset) * task_steps])
                    starting_reward = before_task_rewards[-1]
                    
                    # Training region for subsequent_task_id
                    subsequent_region = [(subsequent_task_id + offset) * task_steps,
                                         (subsequent_task_id + offset + 1) * task_steps]
                    subsequent_task_rewards = self.get_rewards_for_region(xs, ys, subsequent_region)

                    if forgetting_type == 'final':
                        forgetting = starting_reward - subsequent_task_rewards[-1]
                    elif forgetting_type == 'average':
                        forgetting = starting_reward - subsequent_task_rewards.mean()
                    elif forgetting_type == 'worst':
                        forgetting = starting_reward - subsequent_task_rewards.min()
                    else:
                        msg = "Valid forgetting_type values are 'final', 'average', and 'worst'"
                        raise ValueError(msg) 
                        
                    if cycle_id not in per_run_forgetting_per_subsequent[subsequent_task_id]:
                        per_run_forgetting_per_subsequent[subsequent_task_id][cycle_id] = []
                    per_run_forgetting_per_subsequent[subsequent_task_id][cycle_id].append(forgetting)
                    # print(task_id, subsequent_task_id, cycle_id, per_run_forgetting_per_subsequent)
        
        return per_run_forgetting_per_subsequent

    def compute_transfer_metric(
        self, 
        task_results, 
        task_steps, 
        task_id, 
        num_tasks, 
        num_cycles, 
        return_scale,
        forgetting_type='final'
    ) -> Dict[str, Dict[str, List]]:
        """Computes how much reward on `task_id` improves while training on each other task/cycle.

        For each training task and cycle, compares the reward on eval task
        `task_id` right before that training task starts to the reward on
        `task_id` (final/average/worst, per `forgetting_type`) during that
        training task's region.

        Args:
            task_results: Per-run list of (step, value) reward data for `task_id`.
            task_steps: Number of steps allotted per task.
            task_id: The evaluation task whose transfer is being measured.
            num_tasks: Total number of tasks.
            num_cycles: Number of task cycles to consider.
            return_scale: Scale factor applied to raw reward values.
            forgetting_type: 'final', 'average', or 'worst', which reward
                within the training task's region to compare against the
                starting reward.

        Returns:
            Dict mapping training task id to a dict mapping cycle id to a list
            of per-run transfer values for `task_id`.

            Example (for task_id = 0):
                Task 0:
                    Cycle 0: amount of transfer for task ID 0
                    Cycle 1: amount of transfer for task ID 0
                Task 1:
                    Cycle 0: amount of transfer for task ID 0
                    Cycle 1: amount of transfer for task ID 0
                Task 2:
                    Cycle 0: amount of transfer for task ID 0
                    Cycle 1: amount of transfer for task ID 0

        Raises:
            ValueError: If `forgetting_type` is not 'final', 'average', or 'worst'.
        """
        per_training_task = {id: {} for id in range(num_tasks)}  # Inner dict maps cycle to total

        # Loop over runs for experiment
        for run_id, task_result in enumerate(task_results):
            xs = np.array([t[0] for t in task_result])
            ys = np.array([t[1] for t in task_result]) * return_scale

            # Loop over cycles
            for cycle_id in range(num_cycles):
                for training_task_id in range(num_tasks):

                    offset = cycle_id * num_tasks

                    # Before training region
                    if cycle_id == 0 and training_task_id == 0:
                         starting_reward = ys[0]
                    else:
                        before_task_rewards = self.get_rewards_for_region(xs, ys, [None, (training_task_id + offset) * task_steps])
                        starting_reward = before_task_rewards[-1]
                    
                    # Training region 
                    training_region = [(training_task_id + offset) * task_steps,
                                        (training_task_id + offset + 1) * task_steps]
                    training_task_rewards = self.get_rewards_for_region(xs, ys, training_region)

                    if forgetting_type == 'final':
                        transfer = training_task_rewards[-1] - starting_reward
                    elif forgetting_type == 'average':
                        transfer = training_task_rewards.mean() - starting_reward 
                    elif forgetting_type == 'worst':
                        transfer = training_task_rewards.min() - starting_reward 
                    else:
                        msg = "Valid forgetting_type values are 'final', 'average', and 'worst'"
                        raise ValueError(msg) 
                        
                    if cycle_id not in per_training_task[training_task_id]:
                        per_training_task[training_task_id][cycle_id] = []
                    per_training_task[training_task_id][cycle_id].append(transfer)

        return per_training_task
    
    def compute_average_return(
        self, 
        task_results, 
        task_steps, 
        num_tasks, 
        num_cycles, 
        return_scale,
    ) -> Dict[str, Dict[str, List]]:
        """Computes the average reward during each training task/cycle's own region.

        Args:
            task_results: Per-run list of (step, value) reward data for the eval
                task being measured.
            task_steps: Number of steps allotted per task.
            num_tasks: Total number of tasks.
            num_cycles: Number of task cycles to consider.
            return_scale: Scale factor applied to raw reward values.

        Returns:
            Dict mapping training task id to a dict mapping cycle id to a list
            of per-run average-return values.
        """
        per_training_task = {id: {} for id in range(num_tasks)}  # Inner dict maps cycle to total
    
        # Loop over runs for experiment
        for run_id, task_result in enumerate(task_results):
            xs = np.array([t[0] for t in task_result])
            ys = np.array([t[1] for t in task_result]) * return_scale

            # Loop over cycles  
            for cycle_id in range(num_cycles):
                for training_task_id in range(num_tasks):
    
                    offset = cycle_id * num_tasks
                    
                    # Training region 
                    training_region = [(training_task_id + offset) * task_steps,
                                        (training_task_id + offset + 1) * task_steps]
                    training_task_rewards = self.get_rewards_for_region(xs, ys, training_region)

                    avg_return = training_task_rewards.mean()

                    if cycle_id not in per_training_task[training_task_id]:
                        per_training_task[training_task_id][cycle_id] = []
                    per_training_task[training_task_id][cycle_id].append(avg_return)
        
        return per_training_task
    
    def compute_forward_transfer_metric(self, task_results, task_steps, prior_task_ids, return_scale):
        """Computes how much reward on a task is achieved via prior tasks, before it is ever trained.

        For each prior task, compares the reward at the end of that prior task's
        training region to a baseline.

        Args:
            task_results: Per-run list of (step, value) reward data for the
                (not-yet-trained) eval task being measured.
            task_steps: Number of steps allotted per task.
            prior_task_ids: Task ids trained before the eval task, to measure
                forward transfer from.
            return_scale: Scale factor applied to raw reward values.

        Returns:
            Dict mapping prior task id to a list of per-run forward-transfer values.
        """
        per_run_transfer_per_prior = {id: [] for id in prior_task_ids}  # The id maps to task_id, and the entries of the array correspond to separate runs
        
        # Loop over runs for experiment
        for run_id, task_result in enumerate(task_results):
            xs = np.array([t[0] for t in task_result])
            ys = np.array([t[1] for t in task_result]) * return_scale
            
            # Select only the rewards from the region up to and including the training of the given task
            initial_task_value = ys[0]  # TODO: this isn't necessarily a robust average
    
            for prior_task_id in prior_task_ids:
                prior_region = [prior_task_id * task_steps, (prior_task_id+1) * task_steps]  # TODO: could do from the end of the task up to the subsequent one we're looking at...
                subsequent_task_rewards = self.get_rewards_for_region(xs, ys, prior_region)
                last_reward = subsequent_task_rewards[-1]
                baseline = initial_task_value

                if prior_task_id > 0:
                    pre_task_region = [0, prior_task_id * task_steps]  # Get the rewards up to and not including our "previous task"
                    subsequent_pre_task_rewards = self.get_rewards_for_region(xs, ys, pre_task_region)
                    baseline = subsequent_pre_task_rewards[-1]
    
                transfer = last_reward - baseline
                per_run_transfer_per_prior[prior_task_id].append(transfer)

        return per_run_transfer_per_prior

    def get_rewards_for_region(self, xs, ys, region):
        """Selects the y values whose x falls strictly within `region`.

        Args:
            xs: Array of x (step) values.
            ys: Array of y (reward) values, same length as `xs`.
            region: `[lower, upper]` bounds (exclusive); either bound may be
                None to leave that side unbounded.

        Returns:
            The subset of `ys` whose corresponding `xs` fall within `region`.
        """
        # If we have no lower bound specified, all xs are valid
        valid_x_mask_lower = xs > region[0] if region[0] is not None else True  
        valid_x_mask_upper = xs < region[1] if region[1] is not None else True
        valid_x_mask = valid_x_mask_lower * valid_x_mask_upper

        return ys[valid_x_mask]

    def combine_exp_data(self, data, tags):
        """Combines multiple runs of one experiment into a single mean/SEM curve per tag.

        Interpolates each run onto a common, evenly spaced x grid (bounded by the
        overlap of all runs' x ranges) so that runs with differing numbers of
        evaluations can be averaged together, then computes the mean and standard
        error of the mean across runs at each grid point.

        Args:
            data: Dict mapping run id to dict mapping tag to a list of
                (step, value) pairs.
            tags: The tags to combine.

        Returns:
            Dict mapping tag to `[interpolated_xs, y_means, y_stds]`.
        """
        d = {}
        # Loop over train/eval tags
        for k in tags:
            xs = []
            ys = []
            # Loop over multiple runs for same exp to extract x/y axis data again
            for run_id in data.keys():
                run_data = data[run_id][k]
    
                xs.append(np.array([data_point[0] for data_point in run_data]))
                ys.append(np.array([data_point[1] for data_point in run_data]))

            # Get the bounds and the number of samples to take for the interpolation we're about to do
            # We don't try interpolate out of the bounds of what was collected (i.e. below an experiment's min, or above its max)
            min_x = np.array([x.min() for x in xs]).max()
            max_x = np.array(
                [x.max() for x in xs]
            ).min()  # Get the min of the maxes so we're not interpolating past the end of collected data
            num_points = (
                np.array([len(x) for x in xs]).max() * 2
            )  # Doubled from my vague signal processing recollection to capture the underlying signal (...very rough)
    
            # Get the regular interval we'll be interpolating to
            interpolated_xs = np.linspace(min_x, max_x, num_points)
            interpolated_ys_per_run = []

            # Interpolate each run
            for run_id, run_ys in enumerate(ys):
                run_xs = xs[run_id]
                interpolated_ys = np.interp(interpolated_xs, run_xs, run_ys)
                interpolated_ys_per_run.append(interpolated_ys)
            
            y_series = np.array(interpolated_ys_per_run)
            y_means = y_series.mean(0)
            y_stds = sem(y_series)  # Computing the standard error of the mean, since that's what we're actually interested in here.
    
            d[k] = [interpolated_xs, y_means, y_stds]
        return d

    def plot_models(self, d):
        """Renders and saves one Plotly figure per task, overlaying every model's reward curve.

        Args:
            d: Dict mapping model name to that model's combined data (as
                returned by `combine_exp_data`), i.e. dict mapping tag to
                `[xs, y_means, y_stds]`.

        Returns:
            Dict mapping task index to its rendered `plotly.graph_objects.Figure`.
        """
        num_task_steps = self._exp_data['num_task_steps']
        num_cycles = self._exp_data['num_cycles']
        num_tasks = self._exp_data.get('num_tasks', len(self._exp_data['tasks']))
        title_size = self._exp_data['title_size']
        yaxis_label = self._exp_data.get("yaxis_label", "Average Return")
        xaxis_label = self._exp_data.get("xaxis_label", "Steps")
        x_range = [-10, num_task_steps * num_tasks * num_cycles]
        x_scale = self._exp_data['x_scale']
        output_dir = self._exp_data.get('output_dir')
        axis_size = self._exp_data['axis_size']
        axis_label_size = self._exp_data['axis_label_size']
        legend_size = self._exp_data['legend_size']
        legend_entrywidth = self._exp_data['legend_entrywidth']
        exp_name = self._exp_data['exp_name']
        ext = self._exp_data.get('plot_ext', 'pdf')

        figures = {}
        for task_i, task_v in enumerate(self._exp_data['tasks']):
            task_k = task_v['name']
            fig = go.Figure()
            y_range = task_v.get('y_range', None)
            train_regions = task_v.get('train_regions', None)
            showlegend = task_v.get('showlegend', True)
            yaxis_dtick = task_v.get('yaxis_dtick', None)
    
            tag = f"{self._exp_data['tag_base']}/{task_v['i']}"
          
            for model_k, model_v in self._exp_data['models'].items():
                data = d[model_k][tag]
        
                low_trace, trace, up_trace = self.create_scatters(
                    data, 
                    model_k, 
                    model_v, 
                )
    
                fig.add_trace(low_trace)
                fig.add_trace(trace)
                fig.add_trace(up_trace)
         
            yaxis_range = [y_range[0], y_range[1] * 1.01]
            fig.update_layout(
                width=1500,
                height=1200,
                yaxis=dict(
                    title=dict(text=yaxis_label, font=dict(size=axis_label_size), standoff=40),
                    range=yaxis_range,
                    dtick=yaxis_dtick,
                    tickfont=dict(size=axis_size),
                    gridcolor='rgb(230,236,245)',
                    zerolinecolor='rgb(230,236,245)',
                    zerolinewidth=1,
                ),
                xaxis=dict(
                    title=dict(text=xaxis_label, font=dict(size=axis_label_size), standoff=40),
                    tickmode='array',
                    range=x_range,
                    tickvals=self._exp_data.get('xaxis_tickvals', None),
                    ticktext=self._exp_data.get('xaxis_text', None),
                    tickfont=dict(size=axis_size),
                    gridcolor='rgb(230,236,245)',
                    linewidth=1,
                    zerolinewidth=1,
                    automargin=True,                   
                ),
                title=dict(
                    text=task_k, 
                    font=dict(size=title_size),
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                ) if title_size is not None else None,
                legend=dict(
                    font=dict(size=legend_size),
                    xanchor="center",
                    x=0.5,
                    yanchor='top',
                    entrywidth=legend_entrywidth,
                    entrywidthmode='fraction',
                    itemwidth=30,
                    y=-0.25,
                    orientation="h",
                ),
                showlegend=showlegend,
                plot_bgcolor='rgb(255,255,255)',
            )
            fig.update_xaxes(
                showline=True, 
                mirror=True, 
                linewidth=1, 
            )
            fig.update_yaxes(
                showline=True,
                mirror=True,
                linewidth=1,
            )
            if train_regions is not None:
                for shaded_region in train_regions:
                    fig.add_vrect(
                        x0=shaded_region[0],
                        x1=shaded_region[1],
                        fillcolor="rgba(230, 236, 245, 0.5)",
                        layer='below',
                        line_width=0
                    )

            # Use exp_dir and output_path to generate plot/ save dir
            filename = f'{exp_name}-task_{task_i}.{ext}'
            avg_folder = f"{self._exp_data['filter']}{self._exp_data['rolling_mean_count']}"
            save_dir_path = Path(output_dir)/'plots'/avg_folder
            if not save_dir_path.exists():
                save_dir_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(save_dir_path/filename)
            figures[task_i] = fig
            # fig.show()
        
        return figures
    
    def create_scatters(self, data, model_k, model_v, showlegend=True, alpha=None):
        """Builds the three Plotly traces (mean line + shaded std-dev band) for one model's curve.

        Args:
            data: `[x, y_mean, y_std]` as produced by `combine_exp_data`.
            model_k: Model name, used as the trace name/legend group.
            model_v: The model info dict (see `Metrics.__init__`); used for its
                'color', 'color_alpha', and optional 'line' styling.
            showlegend: Whether the mean-line trace should appear in the legend.
            alpha: If given, overrides the line color's alpha channel.

        Returns:
            A list `[lower_bound, trace, upper_bound]` of
            `plotly.graph_objects.Scatter` traces, in the order they should be
            added to the figure (lower bound first, so the fill from the mean
            line shades between it and the lower bound).
        """
        x, y_mean, y_std = data
    
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std
    
        line_color = copy.deepcopy(model_v['color']).replace(" ", "")
        fill_color = copy.deepcopy(line_color).replace(" ", "")
        fill_color = fill_color.replace(',1)', f", {model_v['color_alpha']})")
    
        if alpha is not None:
            line_color = line_color.replace(',1)', f", {alpha})")
     
        plot_name = model_k
        upper_bound = go.Scatter(
            x=x,
            y=y_upper,
            mode='lines',
            line=dict(width=0),
            fillcolor=fill_color,
            fill='tonexty',
            name=plot_name,
            showlegend=False,
        )

        line = model_v.get('line', {})
        if 'color' not in line:
            line['color'] = line_color
        
        trace = go.Scatter(
            x=x,
            y=y_mean,
            mode='lines',
            line=line,
            fillcolor=fill_color,
            fill='tonexty',
            name=plot_name,
            showlegend=showlegend,
            legendgroup=plot_name
        )
    
        lower_bound = go.Scatter(
            x=x, y=y_lower, line=dict(width=0), mode='lines', name=model_k, showlegend=False
        )
    
        # Trace order can be important
        # with continuous error bars
        traces = [lower_bound, trace, upper_bound]
    
        return traces
    
    def plot_metrics(self, metrics: Dict[str, Dict]):
        """Tabulates every metric per model and writes LaTeX/Excel summary tables to disk.

        For each model and each metric (forgetting variants, transfer variants,
        average return, forward transfer), builds a `TabulateMetrics`/
        `TabulateTransferMetrics` table and writes it out as LaTeX (one file per
        metric) and Excel; also writes combined summary tables for the metrics
        named in `self._exp_data['summary_metrics']`, both grouped by metric and
        grouped by task.

        Args:
            metrics: Nested dict of metric data for model, task id, metric, eval
                task, and cycle, as produced by `compute_metrics`.

                Example:
                    Model:
                        Task ID:
                            Forgetting:
                                eval ID:
                                    Cycle: Data
        """
        metric_scale = 10
        transfer_tables = {'forward_transfer': {}}
        metric_keys = {
            'forgetting': {}, 
            'average_forgetting': {}, 
            'worst_forgetting': {},
            'transfer': {},
            'average_transfer': {},
            'worst_transfer': {},
            'average_return': {}
        }
        metric_tables= {}
        metric_tables['latex'] = copy.deepcopy(metric_keys)
        metric_tables['pandas'] = copy.deepcopy(metric_keys)
        
        for model_name, model_metrics in metrics.items():
            for metric in metric_tables['latex'].keys():
                # print(model_name, metric)
                tabulate = TabulateMetrics(
                    task_tags=self.get_metric_tags(),
                    num_cycles=self._exp_data["num_cycles_for_forgetting"],
                    metric_scale=metric_scale if metric != 'average_return' else 1
                )
                
                tabulate(model_metrics=model_metrics, metric_key=metric)
                # print("-----------")
                df, latex = self.generate_metric_table(
                    tabulate.table, 
                    tabulate.error_table,
                    name=metric,
                    negative_as_green=True if 'transfer' not in metric else False,
                    table_caption=f"{model_name}",
                    num_cycles=self._exp_data["num_cycles_for_forgetting"],
                    metric_scale=tabulate.metric_scale
                )
                metric_tables['latex'][metric][model_name] = latex
                metric_tables['pandas'][metric][model_name] = df
                
            # Tabulate Forward Transfer
            tabular_transfer = TabulateTransferMetrics(
                task_tags=self.get_metric_tags(),
                num_cycles=self._exp_data["num_cycles_for_forgetting"],
                metric_scale=metric_scale
            )
            tabular_transfer(model_metrics=model_metrics)
            transfer_tables['forward_transfer'][model_name]  = self.generate_metric_table(
                tabular_transfer.table, 
                tabular_transfer.error_table, 
                name='forward_transfer',
                negative_as_green=False,
                table_caption=f"{model_name}",
                num_cycles=1,
                metric_scale=metric_scale
            )

        latex_pre= "\\begin{table}[H]\n\\tiny\n\centering\n"
        latex_post =  "\\label{}\n\\caption{}\n\\end{{table}}"

        log_latex_tables = lambda f, tables, name:  [f.write(f"%{model} {name} latex: \n\n{latex_pre}{metrics}{latex_post}\n\n") for model, metrics in tables.items()]
        log_path = Path(self._exp_data.get('output_dir'))/'metrics'

        # Create summary of metrics (ignores forward transfer)
        # Loop over metrics
        summary_metrics = {}

        # Group by metrics and save summary tables separtely
        for metric in self._exp_data.get('summary_metrics', []):
            metric_summary = {}
            for t in range(tabular_transfer.num_tasks): # Number of tasks
                assert metric in metric_tables['pandas']
                for name, table in metric_tables['pandas'][metric].items():
                    if name not in metric_summary:
                        metric_summary[name] = []
                    if metric != 'average_return':
                        metric_summary[name].append(table["Avg ± SEM"].values[t])
                    else:
                        metric_summary[name].append(table.loc["Avg ± SEM", :].values[t])

                sm_df = pd.DataFrame.from_dict(metric_summary, orient='index')
                with open(log_path/f"summary_{metric}_latex.txt", "w") as f:
                    f.write(f"%{' '.join(list(metric_summary.keys()))} latex: \n\n{sm_df.to_latex()}\n\n")

        # Group by task and save a single summary table
        for t in range(tabular_transfer.num_tasks):
            for metric in self._exp_data.get('summary_metrics', []):
                assert metric in metric_tables['pandas']
                # Loop over algos
                for name, table in metric_tables['pandas'][metric].items():
                    if name not in summary_metrics:
                        summary_metrics[name] = []
                    if metric != 'average_return':
                         summary_metrics[name].append(table["Avg ± SEM"].values[t])
                    else:
                        summary_metrics[name].append( table.loc["Avg ± SEM", :].values[t])
        if len(summary_metrics) > 0:
            sm_df = pd.DataFrame.from_dict(summary_metrics, orient='index')
            with open(log_path/"summary_metrics_latex.txt", "w") as f:
                f.write(f"%{' '.join(list(summary_metrics.keys()))} latex: \n\n{sm_df.to_latex()}\n\n")
    
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        for name, table in metric_tables['latex'].items():
            with open(log_path/f"{name}_latex.txt", "w") as f:
                log_latex_tables(f, table, name)
        for name, trans_table in transfer_tables.items():
            with open(log_path/f"{name}_latex.txt", "w") as f:
                log_latex_tables(f, trans_table, name)        
            
    def generate_metric_table(
        self, 
        metric_table, 
        metric_error_table, 
        negative_as_green, 
        table_caption, 
        num_cycles, 
        metric_scale, 
        name,
    ):
        """Formats a metric table into a styled LaTeX table and pandas DataFrame, and saves it to Excel.

        Args:
            metric_table: 2D array of metric values (rows = tasks + grand average,
                columns = task/cycle combos + grand average).
            metric_error_table: 2D array of the corresponding SEM values, same
                shape as `metric_table`.
            negative_as_green: If True, negative values are colored green (and
                positive red); if False, the reverse. Used to indicate "good"
                vs. "bad" direction differently for forgetting vs. transfer.
            table_caption: Caption used for the LaTeX table and the saved
                Excel filename.
            num_cycles: Number of cycles represented in the table's columns.
            metric_scale: Scale factor the metric values were multiplied by
                (used to normalize the color-intensity mixin).
            name: Metric name, used as the subdirectory for the saved Excel file.

        Returns:
            Tuple `(df, latex_metrics)`: the table as a pandas DataFrame (row per
            task, transposed) and as a LaTeX string with color-coded cells.
        """
        def style_forgetting_table(v):
            default_mixin_val = 40
    
            # Mixin => how much of the color (vs how much white)
            v = "--" if v == "--" else float(v.split("±")[0])  # Undo the SEM inclusion. A bit hacky but whatever
            mixin_val = 0 if v == '--' else int(np.abs(v) * default_mixin_val/metric_scale)
            if v == '--':
                color = "green"  # Doesn't matter
            elif (not negative_as_green and v > 0) or (negative_as_green and v < 0):
                color = "green"
            else:
                color = "red"
    
            return f"cellcolor:{{{color}!{mixin_val}}}"  # Exclamation point is a mixin - says how much of the given color to use (mixed in with white)
    
        task_names = [t['name'] for t in self._exp_data["tasks"]]
        tasks = self.truncate_task_names(task_names, max_len=20)
    

        col_names = [f"T{x+1}-C{c+1}" for c in range(num_cycles) for x in range(len(tasks))]
        col_names += ["Avg ± SEM"]
        row_names = [f"T{x+1}" for x in range(len(tasks))] + ["Avg ± SEM"]
    
        # Convert to string and include the error metric
        string_metric_table = np.array(metric_table, dtype=object)
        for i in range(len(metric_table)):
            for j in range(len(metric_table[0])):
                if not np.isnan(metric_table[i][j]):
                    string_metric_table[i][j] = f"{metric_table[i][j]:.2f} ± {metric_error_table[i][j]:.2f}"
                else:
                    string_metric_table[i][j] = "--"
    
        # Styling for Latex isn't quite the same as other formats, see: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
        df = pd.DataFrame(string_metric_table)
        df = df.rename(columns=lambda x: col_names[x])  # Name the columns: "Task Name (C cycle_id)"
        df = df.rename(index=lambda x: row_names[x])  # Name the rows: "Task Name"
        df = df.T

        #data_style = df.style.format(precision=1, na_rep="--")
        data_style = df.style.applymap(style_forgetting_table)
        data_style = data_style.set_table_styles([
            {'selector': 'toprule', 'props': ':toprule;'},
            {'selector': 'bottomrule', 'props': ':bottomrule;'},
        ], overwrite=False)
    
        column_style = ''.join(['P{1.08cm}' for _ in range(len(data_style.columns) - 1)])
        latex_metrics = data_style.to_latex(column_format=f"P{{0.6cm}}|{column_style}|P{{1.08cm}}") 
        csv_path = Path(self._exp_data.get('output_dir'))/'metrics'/name
        if not csv_path.exists():
            csv_path.mkdir(parents=True, exist_ok=True)
        data_style.to_excel(str(csv_path/f'{table_caption}.xlsx'))
    
        return df, latex_metrics

    def truncate_task_names(self, task_names, max_len):
        """Truncates each task name to `max_len` characters, appending '..' if cut.

        Args:
            task_names: List of task name strings.
            max_len: Maximum length to keep before truncating.

        Returns:
            A new list of (possibly truncated) task names.
        """
        new_task_names = []
        for task_name in task_names:
            if len(task_name) > max_len:
                new_task_name = task_name[:max_len] + ".."
            else:
                new_task_name = task_name
    
            new_task_names.append(new_task_name)
    
        return new_task_names