
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from pathlib import Path
from typing import Callable, Any, List, Tuple
from itertools import groupby
import numpy as np
from design_util import find_index_by_config_id, get_subdirectories, make_output_dir
import json
import pickle
from statsmodels.nonparametric.smoothers_lowess import lowess
import os


absolut_parameters = ["tower_yValue", "tower_xValue"]

def dynamic_visualization(called_function: Callable[..., Any], name:str, run: str, output_path: str, *args: Any):
    '''
    visualize the given function with the given arguments and save it to the output path
    is mainly used for the hpbandster.visualization functions
    '''
    # Call the provided function with the given arguments
    fig, ax = called_function(*args)
    current_title = plt.gca().title.get_text()
    current_suptitle = fig._suptitle
    if current_title is "":
        current_title = called_function.__name__
    if current_suptitle is not None:
        plt.suptitle(f"{current_suptitle.get_text()} \n (run {run})")
    else:
        # Set the updated title
        plt.title(f"{current_title} \n (run {run})")
    plt.savefig(f'{output_path}/{name}.png')




def get_best_config(run_path: str) -> Tuple[dict, float]:
    '''
    returns the best configuration of the run
    return
        - dict: best configuration
        - float: loss of the best configuration
    '''
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_path)
    all_runs = result.get_all_runs()

    all_runs =  [d for d in all_runs if d["info"] is not None]
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    inc_index = find_index_by_config_id(all_runs, inc_id)
    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_budget = inc_run.info['budget']

    print('Best configuration:')
    print(inc_config)
    print('It achieved loss of %f (evaluation) with budget %f. --> (ID: %i)'%(inc_loss, inc_budget, inc_index))
    return inc_config, inc_loss


def viusualize_run(run_path: str, output_path: str):
    '''
    plot all the hpbandster visualizations for the given run
    '''
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_path)
    # get all executed runs
    all_runs = result.get_all_runs()
    all_runs =  [d for d in all_runs if d["info"] is not None]
    run_path = Path(run_path)

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    plotid = 1
    dynamic_visualization(hpvis.losses_over_time,f"plot_{plotid}", run_path.name, output_path, all_runs)
    plotid+=1
    dynamic_visualization(hpvis.concurrent_runs_over_time,f"plot_{plotid}", run_path.name, output_path, all_runs)
    plotid+=1
    dynamic_visualization(hpvis.finished_runs_over_time,f"plot_{plotid}", run_path.name, output_path, all_runs)
    plotid+=1
    dynamic_visualization(hpvis.correlation_across_budgets,f"plot_{plotid}", run_path.name, output_path, result)
    # plotid+=1
    # dynamic_visualization(hpvis.performance_histogram_model_vs_random,f"plot_{plotid}", run_path.name, output_path, all_runs, id2conf)


def interactive_vis(run_path: str, output_path: str):
    '''
    hpbandster interactive visualization function
    https://automl.github.io/HpBandSter/build/html/auto_examples/plot_example_7_interactive_plot.html#sphx-glr-auto-examples-plot-example-7-interactive-plot-py
    '''
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_path)

    # get all executed runs
    all_runs = result.get_all_runs()
    all_runs =  [d for d in all_runs if d["info"] is not None]

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    lcs = result.get_learning_curves()

    hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))
    plt.savefig(f'scripts/logs/design/plots/plot_interactive.png')
    def realtime_learning_curves(runs):
        """
    	example how to extract a different kind of learning curve.

    	The x values are now the time the runs finished, not the budget anymore.
    	We no longer plot the validation loss on the y axis, but now the test accuracy.

    	This is just to show how to get different information into the interactive plot.

    	"""
        sr = sorted(runs, key=lambda r: r.budget)
        lc = list(filter(lambda t: not t[1] is None, [(r.time_stamps['finished'], r.info['budget']) for r in sr]))
        return([lc,])

    lcs = result.get_learning_curves(lc_extractor=realtime_learning_curves)

    hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))
    plt.savefig(f'{output_path}/plot_interactive_realtime.png')


def config_losses(run_path: str, output_path: str, inc_config: dict, inc_loss: float, run_path_mpbt: str = None) -> None:
    '''
    plot the loss vs the configuration of the run
    '''
    # Create figure and subplots
    result = hpres.logged_results_to_HBS_result(run_path)
    # get all executed runs
    all_runs = result.get_all_runs()
    if run_path_mpbt is not None:
        result_mpbt = hpres.logged_results_to_HBS_result(run_path_mpbt)
        all_mpbt_runs = result_mpbt.get_all_runs()


    all_runs =  [d for d in all_runs if d["info"] is not None]
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    config_strings = [key for key,_ in inc_config.items()]
    fig, axes = plt.subplots(nrows=len(config_strings), ncols=1, figsize=(14, 16))
    # Find unique budget values
    unique_budgets = set(int(d['budget']) for d in all_runs)
    for entry in all_runs:
        config_id = entry['config_id']
        if config_id in id2conf:
            model_based_pick = id2conf[config_id]["config_info"]['model_based_pick']
            entry['info']['marker'] = 'o' if model_based_pick else '^'
            entry['info']['marker_string'] = 'Model-Based' if model_based_pick else 'Random'
        else:
            ['info']['marker'] = 'o'  # Default to circle if config_id not found
            entry['info']['marker_string'] = 'Model-Based'

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, config_str in enumerate(config_strings):
        degrees = False if config_str in absolut_parameters else True
        for color_idx, budget in enumerate(unique_budgets):
            budget_data = [(d['info'][config_str], d['loss'], d["info"]['marker']) for d in all_runs if int(d['budget']) == budget]
            # Sort the array based on the last element of each tuple
            sorted_array = sorted(budget_data, key=lambda x: x[-1])

            # Use groupby to group the tuples based on the last element
            marker_seperated = [list(group) for _, group in groupby(sorted_array, key=lambda x: x[-1])]
            for marker_array in marker_seperated:
                design_values, loss_values, markers = zip(*marker_array)
                if degrees:
                    design_values = np.rad2deg(design_values)

                marker_string = "Model-Based" if markers[0] == "o" else "Random"
                axes[idx].scatter(design_values, loss_values, marker=markers[0], label=f"Budget {budget}, {marker_string}", color=colors[color_idx])

        if run_path_mpbt is not None:
            run_data = [(run['info'][config_str], run['loss']) for run in all_mpbt_runs]
            config_values, loss_values = zip(*run_data)
            if degrees:
                config_values = np.rad2deg(config_values)
            axes[idx].scatter(config_values, loss_values, marker='d', label="manipulability", color=colors[len(config_strings)], s=30)

        best_config = np.rad2deg(inc_config[config_str]) if degrees else inc_config[config_str]
        axes[idx].scatter(best_config, inc_loss, facecolors='none', edgecolors='g' , label="best config", s=120)
        axes[idx].set_title(config_str)
        if degrees:
            axes[idx].set_xlabel('degree')
        else:
            axes[idx].set_xlabel('m from origin')
        axes[idx].set_ylabel('Loss')
        axes[idx].legend(title='Meaning', bbox_to_anchor=(1, 1), loc='upper left')
    suptitle_str = f'Config Loss Relation \n run {Path(run_path).parent.name}'
    if run_path_mpbt is not None:
        suptitle_str += f", mpbt {Path(run_path_mpbt).name}"
    plt.suptitle(suptitle_str)
    # Adjust layout to include legend in saved image
    plt.tight_layout()
    plt.savefig(f'{output_path}/config_losses.png', bbox_inches="tight")




def designs_to_losses(run_path: str, output_path: str) -> Tuple[plt.Figure, plt.Axes]:
    '''
    plot the design vs the loss of the run
    deprecated since i use more than one design parameter
    but kept here if i have the results of an older run i want to visualize
    '''
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_path)
    run_path = Path(run_path)
    # get all executed runs
    all_runs = result.get_all_runs()

    all_runs =  [d for d in all_runs if d["info"] is not None]
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Find unique budget values
    unique_budgets = set(int(d['budget']) for d in all_runs)
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,6))
    # resolve model_based property
    for entry in all_runs:
        config_id = entry['config_id']
        if config_id in id2conf:
            model_based_pick = id2conf[config_id]["config_info"]['model_based_pick']
            entry['info']['marker'] = 'o' if model_based_pick else '^'
            entry['info']['marker_string'] = 'Model-Based' if model_based_pick else 'Random'
        else:
            ['info']['marker'] = 'o'  # Default to circle if config_id not found
            entry['info']['marker_string'] = 'Model-Based'

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Plotting
    for idx, budget in enumerate(unique_budgets):
        # budget_data = [(d['info']['design'], d['loss'], d["info"]['marker']) for d in all_runs if d['budget'] == budget]
        budget_data = [(d['info']['arm_pitch'], d['loss'], d["info"]['marker']) for d in all_runs if d['budget'] == budget]
        # Sort the array based on the last element of each tuple
        sorted_array = sorted(budget_data, key=lambda x: x[-1])

        # Use groupby to group the tuples based on the last element
        marker_seperated = [list(group) for key, group in groupby(sorted_array, key=lambda x: x[-1])]
        for marker_array in marker_seperated:
            design_values, loss_values, markers = zip(*marker_array)
            design_values_degrees = np.degrees(design_values)

            marker_string = "Model-Based" if markers[0] == "o" else "Random"
            plt.scatter(design_values_degrees, loss_values, marker=markers[0], label=f"Budget {budget}, {marker_string}", color=colors[idx])
    # Customize the plot
    plt.xlabel('Design (Degrees)')
    plt.ylabel('Loss')
    plt.legend(title='Meaning', bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(f'Loss vs. Design for Different Budgets \n run {run_path.name}')
    # Adjust layout to include legend in saved image
    plt.tight_layout()
    plt.savefig(f'{output_path}/design_losses.png', bbox_inches="tight")
    return (fig, ax)


def show_metric_of_run(ray_run_path: str, run_number: int, output_path: str, metrics: List[str], custom_metric: bool = True) -> None:
    '''
    plot the given metric of the run
    run_number starts with 1
    '''
    sub_dirs = get_subdirectories(ray_run_path)
    # assert run_number <= len(sub_dirs)
    result_path = f"{ray_run_path}/{run_number}"
    sub_dirs = get_subdirectories(result_path)
    result_path = f"{result_path}/{sub_dirs[0]}/result.json"
    # Read each line and parse it separately
    whole_json = []
    with open(result_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                # 'data' now contains the parsed JSON content of the current line
                whole_json.append(data)
                # print(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}. Error: {e}")
    def format_ticks(value, _):
        return f"{int(value // 1000)}K"
    fig, ax = plt.subplots()
    timesteps = [1000 * i for i in range(len(whole_json))]
    if custom_metric:
        whole_json = [item.get("custom_metrics") for item in whole_json]

    for metric in metrics:
        metric_values = [item.get(metric) for item in whole_json if metric in item]
        ax.plot(timesteps, metric_values, label=metric)
    plt.title(f"Evaluation of run {Path(ray_run_path).name}")
    # Add labels and legend
    # Apply the custom formatter to the x-axis
    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.xlabel("Timesteps")
    plt.ylabel("")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_path}/{metric}.png', bbox_inches="tight")

def filter_tuples_by_value(tuples_list, value):
    return list(map(lambda t: t[0], filter(lambda t: t[1] == value, tuples_list)))


def show_evaluation_results(run_path: str, output_path: str, inc_config: dict) -> None:
    '''
    plot the performance on the evaluation tasks of the run
    '''
    result = hpres.logged_results_to_HBS_result(run_path)
    run_path = Path(run_path)
    # get all executed runs
    all_runs = result.get_all_runs()
    # undo runs which ends in errors
    all_runs =  [d for d in all_runs if d["info"] is not None]
    evaluation_array= [(d["info"]["evaluation"], d["budget"]) for d in all_runs]
    # design_values= [(d["info"]["design"], d["budget"]) for d in all_runs]

    budget_values= [d["budget"] for d in all_runs]

    # Find unique budget values
    unique_budgets = set(budget_values)

    # Create subplots for each task
    # tasks = ['rndstartrndgoal', 'picknplace', 'door', 'drawer','simpleobstacle']
    tasks = ['simpleobstacle']
    # Create a figure and a grid of subplots
    max_rows = 3
    max_columns = 2
    config_strings = [key for key,_ in inc_config.items()]
    for config_str in config_strings:
        degrees = False if config_str == "tower_yValue" else True
        fig, axs = plt.subplots(nrows=max_rows, ncols=max_columns, figsize=(12, 12))
        for task_index in range(len(tasks)):
            row = int(task_index / max_columns)
            column = task_index % max_columns
            if column == 0:
                axs[row, column].set_ylabel("Task Success Rate [%]")
            if row == max_rows -1:
                x_label = "Design in Degree" if degrees else "m from origin"
                axs[row, column].set_xlabel(x_label)
            if degrees:
                axs[row, column].set_xlim((0,92))
            axs[row, column].set_ylim((0,100))
            for budget in unique_budgets:
                design_values= [(d["info"][config_str], d["budget"]) for d in all_runs]
                task_success = [x[task_index]*100 for x in (filter_tuples_by_value( evaluation_array, budget))]
                design_values_budget = filter_tuples_by_value(design_values, budget)
                if degrees:
                    design_values_budget = np.degrees(design_values_budget)
                axs[row, column].scatter(design_values_budget, task_success , label=int(budget))
                axs[row, column].legend(title="Training Steps")
                axs[row, column].set_title(tasks[task_index])
        fig.delaxes(axs[2,1])
        # Adjust layout for better spacing
        plt.subplots_adjust(hspace=0.3)  # Increase the vertical space between rows
        plt.suptitle(f"Success Rate for Different Tasks \n config parameter: {config_str}")
        plt.savefig(f'{output_path}/evaluation_tasks_{config_str}.png', bbox_inches="tight")


def get_best_runs(run_path: str, num_runs: int):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_path)
    all_runs = result.get_all_runs()
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()
    sorted_runs = sorted(all_runs, key=lambda x: x.info['avg_manipulability'], reverse=True)
    best_runs = sorted_runs[:num_runs]
    best_run_ids = [all_runs.index(run) for run in best_runs]
    # let's grab the run on the highest budget
    print('Best configuration:')
    best_configs = []
    for idx, run in enumerate(best_runs):
        id = run.config_id
        config = id2conf[id]['config']
        best_configs.append(config)
        print(f"{id}: {run.loss} \n configuration: {config}")
        load_manipulability_3d('manipulability3d', Path(run_path).parent, best_run_ids[idx] + 1 )
    return best_runs, best_configs

def config_losses_mpbt(run_path: str, output_path: str, best_configs: List[dict], best_values: List[float]):
    result = hpres.logged_results_to_HBS_result(run_path)
    all_runs = result.get_all_runs()
    avg_manipulability_border = 0.22
    config_strings = [key for key,_ in best_configs[0].items()]
    fig, axes = plt.subplots(nrows=len(config_strings), ncols=1, figsize=(14, 16))
    for idx, config_str in enumerate(config_strings):
        degrees = False if config_str in absolut_parameters else True
        # run_data = [(run['info'][config_str], run['info']['avg_manipulability']) for run in all_runs]
        # Filter out runs with avg_manipulability below the border
        filtered_run_data = [
            (run['info'][config_str], run['info']['avg_manipulability'])
            for run in all_runs
            if run['info']['avg_manipulability'] >= avg_manipulability_border
        ]
        config_values, loss_values = zip(*filtered_run_data)
        if degrees:
            config_values = np.rad2deg(config_values)
        axes[idx].scatter(config_values, loss_values, label="manipulability",  s=30)

        best_config_value = [best_config[config_str] for best_config in best_configs]
        if degrees:
            best_config_value = np.rad2deg(best_config_value)
        axes[idx].scatter(best_config_value, best_values, facecolors='none', edgecolors='g' , label="best config", s=120)
        # for i, best_config in enumerate(best_configs):
        axes[idx].set_title(config_str)
        if degrees:
            axes[idx].set_xlabel('degree')
        else:
            axes[idx].set_xlabel('m from origin')
        axes[idx].set_ylabel('avg manipulability')
        axes[idx].legend(title='Meaning', bbox_to_anchor=(1, 1), loc='upper left')
    suptitle_str = f'Config Manipulability Relation \n run {Path(run_path).name}'
    plt.suptitle(suptitle_str)
    # Adjust layout to include legend in saved image
    plt.tight_layout()
    plt.savefig(f'{output_path}/config_manipulability.png', bbox_inches="tight")


def visualize_all(filepath: str, ray_run_path: str, best_run_id: int, output_path: str, manipulability: str = None) -> None:
    '''
    container function that runs all single visualization functions toghether
    '''
    make_output_dir(output_path=output_path)
    inc_config, inc_loss = get_best_config(filepath)
    config_losses(filepath, output_path, inc_config, inc_loss, manipulability)
    show_evaluation_results(run_path=filepath, output_path=output_path, inc_config=inc_config)
    viusualize_run(run_path=filepath, output_path=output_path)
    # interactive_vis(run_path=filepath, output_path=output_path)
    # designs_to_losses(run_path=filepath, output_path=output_path)
    show_metric_of_run(ray_run_path, best_run_id, output_path, ["success_nojumps_mean", "success_mean"])
    show_metric_of_run(ray_run_path, best_run_id, output_path, ["success_mean"])

def load_manipulability_3d(folder_name: str, log_path: Path, index: int) -> None:

    figx = pickle.load(open(log_path / folder_name / f'Configuration{index}.pkl', 'rb'))
    figx.show()


def visualize_results(filepath, output_path) -> None:
    result = hpres.logged_results_to_HBS_result(filepath)
    # get all executed runs
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()
    unique_budgets = set(d['budget'] for d in all_runs)
    colors = ["blue", "green","purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    lightcolors = ["skyblue", "lightgreen","purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    # Assuming your list of objects is called 'objects_list'
    # Extract loss values from each object's info
    loss_values_total = [obj.loss for obj in all_runs if obj.loss is not None]
    smoothness_fraction = 0.3
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    
    # Generate y-axis (index of each object, assuming they are ordered by time)
    for i, budget in enumerate(unique_budgets):
        losses, y_value = [], []
        for run_id, run in enumerate(all_runs):
            if run.loss is not None and run.budget == budget:
                losses.append(run.loss)
                y_value.append(run_id)
        
        plt.scatter(y_value,losses,color=lightcolors[i], label=f'Designs at Budget {int(budget)}')
        plt.plot(y_value, losses, color=lightcolors[i], linewidth=1)
        # Perform LOWESS smoothing to fit a smooth curve (fraction controls smoothness)
        smoothed = lowess(endog=losses, exog=y_value, frac=smoothness_fraction)
        # Plot the smooth curve
        plt.plot(smoothed[:, 0], smoothed[:, 1], color=colors[i], linewidth=2)

    y_axis_total = range(len(loss_values_total))
    # Perform LOWESS smoothing to fit a smooth curve (fraction controls smoothness)
    smoothed = lowess(endog=loss_values_total, exog=y_axis_total, frac=smoothness_fraction)

    # Plot the smooth curve
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)

    # Add labels and title
    plt.ylabel('Evaluation Loss (SO)')
    plt.xlabel('Sampled Design over Time')
    plt.title('Loss over Designs of FMM (franka arm)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_path}_loss_over_time.png', bbox_inches="tight")
    # Invert y-axis so the first object is at the top
    # plt.gca().invert_yaxis()
    
    

def visualize_manipulability(filepath, output_path) -> None:
    '''
    container function that runs visualization functions for the manipulability optimization
    '''
    make_output_dir(output_path=output_path)
    best_runs, best_configs = get_best_runs(filepath, 3)
    config_losses_mpbt(filepath, output_path, best_configs, [run.info['avg_manipulability'] for run in best_runs])
    # inc_config, inc_loss = get_best_config(filepath)
    # viusualize_run(run_path=filepath, output_path=output_path)
    # config_losses(filepath, output_path, inc_config, inc_loss)
    
relevant_runs = ["20240226_2311", 
                            "20240710_1215", 
                            "20240710_1215_more_values", 
                            "20240830_0815",
                            "20240831_0849",
                            "20240308_0953",
                            "20240808_0824",
                            "20240812_1215",
                            "20240820_0624",
                            "20240923_1700",
                            "20241017_0659",
                            "20241011_1208",
                            "20241004_0704",
                            # "20240604_2010", #manipulability omni
                            # "20240809_0605", #manipulability diff
                            # "20240924_0744", #manipulability ur5
]

if __name__ == "__main__":
    for run in relevant_runs:
        # run = "20241004_0704"
        output_path_base = "scripts/logs/design/plots"
        bohp_path_old = "scripts/logs/design/bohb/"
        bohp_path = "scripts/logs/design/"
        ray_path = "scripts/logs/ray/design_evaluation_"
        output_path_result_time = f"{output_path_base}/res_time"
        output_path = f"{output_path_base}/{run}"
        # output_path_mpbt = f"{output_path_base}/{run_mpbt}"
        filepath = (bohp_path+run+"/bohb")
        filepath_old = (bohp_path_old+run)
        ray_run_path = (ray_path+run)
        best_run_id = 4
        # manipulability_path = (bohp_path + run_mpbt + "/bohb")
        if not os.path.exists(output_path_result_time):
            os.makedirs(output_path_result_time)
        output_path_result_time = f"{output_path_result_time}/{run}"
        visualize_results(filepath, output_path_result_time)
    # load_manipulability_3d("20240417_1953/manipulability3d",Path(bohp_path),1 )
    # visualize_all(filepath=filepath, ray_run_path=ray_run_path, best_run_id=best_run_id, output_path=output_path)#, manipulability=manipulability_path)

    # visualize_manipulability(manipulability_path, output_path_mpbt)
