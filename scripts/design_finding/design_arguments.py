import argparse
from pathlib import Path


""" 
    This class is used to store the arguments for the design finding script.
"""


class DesignArguments(argparse.Namespace):
    training_steps: int
    resume: bool
    log_path: Path
    # population_size: int
    # rand_start_cnt: int
    # mutation_cnt: int
    # metric: str
    max_budget: float
    min_budget: float
    n_iterations: int
    num_samples: int
    nic_name: str
    run_id: str
    min_points_in_model: int
    bandwidth_factor: int
    eta: float
    n_workers: int
    arm_selection: str
    previous_run_dir: str  # continue form previous run
    optimization_type: str
    manipulability_design_configs_path: str
    use_manipulability_config: str
    
