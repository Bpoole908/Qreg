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