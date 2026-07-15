from continual_rl.experiments.experiment import Experiment

def exp_loader(
    task_func,
    game_names,
    task_prefix=None,
    task_func_kwargs=None,
    exp_kwargs=None
):
    """Builds a `continual_rl` `Experiment` from a list of games by constructing one task per game.

    Args:
        task_func: Callable that builds a single task, called as
            `task_func(task_id, action_space_id, game_name, **kwargs)` (e.g.
            `crl.experiments.make_ple.get_single_ple_task`).
        game_names: List of game/environment names, one task built per entry.
        task_prefix: Optional string prefix prepended (with an underscore) to
            each task's id; if None, the task id is just its action space id.
        task_func_kwargs: Optional list of per-task kwargs dicts, matched
            positionally with `game_names`. If None, every task gets no extra kwargs.
        exp_kwargs: Optional kwargs forwarded to the `Experiment` constructor.

    Returns:
        A `continual_rl.experiments.experiment.Experiment` built from the
        constructed tasks.
    """
    task_func_kwargs = [{}] if task_func_kwargs is None else task_func_kwargs
    exp_kwargs = {} if exp_kwargs is None else exp_kwargs

    tasks = [
        task_func(
            action_space_id if task_prefix is None else f"{task_prefix}_{action_space_id}",
            action_space_id,
            name,
            **kwargs
        ) for action_space_id, (name, kwargs) in enumerate(zip(game_names, task_func_kwargs))
    ]

    return Experiment(tasks, **exp_kwargs)
