import numpy as np
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import pickle
from subprocess import Popen
import time
import csv
import os
from modulation.utils import launch_ros
from evaluation_ray import register_envs_models

from design_finding.design_arguments import DesignArguments
from design_finding.xacro_handler import XacroHandler
from design_finding.design_util import make_output_dir
from design_finding.optimizer_base import OptimizerInterface

class MinimalManipulabilityAnalysis(OptimizerInterface):
    def __init__(self, 
                 logger: logging.Logger, 
                 wandb_config:dict, 
                 main_path:str, 
                 env) -> None:
        super().__init__(logger)
        self.position_grid = self.make_position_gird()
        # [roll, pitch, yaw]
        # associations: cat, pitcher, door
        self.rotation_options = [
            [90, 0, 0],
            [0 , 0, 0],
            [0 ,90, 0],
            [90 ,90, 0],
            [0 ,0, 90],
            [0 ,0, -90],
        ]
        self.manipulability_rot_avg = []
        self.env = env
        self.wandb_config = wandb_config
        self.main_path = main_path
        self.process = None
        self.folder_name = 'manipulability3d'
        self.index = 1
    
    def make_position_gird(self) -> np.ndarray:
        '''
        generate a normal distributet grid of positions for the robot to reach
        '''
        x_min, x_max, x_quant = -0.2, 1.1, 0.1
        y_min, y_max, y_quant = -0.8, 0.8, 0.1
        z_min, z_max, z_quant = 0.1, 1.7, 0.1
        # Generate grid points along each axis
        x_points = np.arange(x_min, x_max + x_quant, x_quant)
        y_points = np.arange(y_min, y_max + y_quant, y_quant)
        z_points = np.arange(z_min, z_max + z_quant, z_quant)

        # Generate meshgrid
        xx, yy, zz = np.meshgrid(x_points, y_points, z_points)

        # Reshape the meshgrid to get positions in the format (x, y, z)
        positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        print(f"positions: {len(positions)}")
        return positions

    def evaluate_design_simple(self) -> Tuple[float, float]:
        '''
        main method called to evaluate a configuration
        '''
        self.relaunch_ros()
        self.env.reload_robot_description()
        time.sleep(3)

        # calculate manipulability for node
        np.random.seed(self.wandb_config.seed)
        avg_manipulability = self.get_manipulability_avg(1)
        self.logger.info(f"avg manipulability: {avg_manipulability}")
        self.index += 1
        return avg_manipulability
    
    def evaluate_design(self):
         print(NotImplementedError("This method is not implemented yet"))
    
    def relaunch_moveit(self):
        # First, stop the existing MoveIt! nodes
        Popen(['rosnode', 'kill', '/move_group'])  # Adjust node name as needed
        Popen(['rosnode', 'kill', '-a'])  # Adjust node name as needed
        # # Then, restart MoveIt! nodes
        Popen(['roslaunch', 'modulation_rl', 'fmm_analytical.launch', 'gui:=false', 'BIOIK:=true'])  # Adjust package and launch file names
        Popen(['rosparam', 'load', '/robot_description'])  # Reload robot description parameters

    def relaunch_ros(self):
        '''
        relaunch the ros nodes
        '''

        if self.process is not None:
            self.process.terminate()
      
        self.process = launch_ros(main_path=self.main_path, config=self.wandb_config,
                                  always_relaunch = True, task=self.wandb_config.task)
        register_envs_models()

    def get_manipulability_avg(self, sampling: int) -> float:
        ''''
        Calculate the average manipulability for a given serial chain and a given number how many times to sample the joint values in beginning
        '''
        self.manipulability_rot_avg = []
        reachability_threshold = 0.1
        ik_float = 1.0
        ik_bool = False
        position_manipulabilities = []
        rot_possible = np.zeros(len(self.rotation_options))
        for position in self.position_grid:
            rot_manipulabilities = []
            for rot_id, rotation in enumerate(self.rotation_options):
                target = position.tolist() + rotation
                sample_mpbt = 0
                for idx in range(sampling):
                    possible = self.env.get_find_ik_python(target, ik_float, ik_bool, reachability_threshold)
                    if possible:
                        mpbt = self.env.get_manipulability('whole_body')
                        if mpbt < 0:
                            print("Error computing the manipulability: pose not accessible")
                            print(f"target: {target}, manipulability: {mpbt}")
                            return -1
                        rot_possible[rot_id] += 1
                        sample_mpbt += mpbt
                    rot_manipulabilities.append(sample_mpbt/sampling)
            avg_rot_manipulabilities = np.average(rot_manipulabilities)
            self.manipulability_rot_avg.append(avg_rot_manipulabilities)
            position_manipulabilities.append((position, avg_rot_manipulabilities))
        result = np.average(self.manipulability_rot_avg)
        self.save_to_csv(position_manipulabilities)
        self.logger.debug(f"manipulability: {result}")
        return result
    

    def plot_manipulability_3d(self, vis: bool = False) -> None:
        
        colors = np.zeros((len(self.manipulability_rot_avg), 3))
        # max_value = max(self.manipulability_rot_avg)
        max_value = 0.6
        for i, avg in enumerate(self.manipulability_rot_avg):
            # interpolate 
            interpolated_avg = np.interp(avg, [0, max_value], [0, 1])
            r = 1 - interpolated_avg  # Red component decreases as average value increases
            g = interpolated_avg      # Green component increases as average value increases
            colors[i] = [r, g, 0]  # No blue component

        # Plotting
        fig = plt.figure() # Adjust figsize as needed
        gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])  # 1 row, 2 columns, width ratio of 8:1
        
        # Plotting 3D scatter plot
        ax = fig.add_subplot(gs[0], projection='3d')
        fig.suptitle('Manipulability 3D')
        ax.set_xlabel('X (depth)')
        ax.set_ylabel('Y (width)')
        ax.set_zlabel('Z (height)')
        # Scatter plot
        x, y, z = self.position_grid[:,0],self.position_grid[:,1],self.position_grid[:,2]
        # Create a mask to remove the quadratic piece
        half_x = np.round(np.min(x) + ((np.max(x) -np.min(x))/ 2), 2)
        half_y = 0
        half_z = np.round(np.min(z) + ((np.max(z) -np.min(z))/ 2), 2)
        mask = (x > half_x) & (y <= half_y) & (z > half_z)
        
        # Filter out the points using the mask
        x_filtered = x[~mask]
        y_filtered = y[~mask]
        z_filtered = z[~mask]
        colors_filtered = colors[~mask]

        ax.scatter(x_filtered, y_filtered, z_filtered, c=colors_filtered)
        # Create a ScalarMappable object
        norm = plt.Normalize(0, max_value)  # Normalizing color values
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        sm.set_array([])  # Empty array required for ScalarMappable

        # Create colorbar
        cbar_ax = fig.add_subplot(gs[1])  # Colorbar axes
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Manipulability')

        plt.tight_layout()  # Adjust layout
        # ax.scatter(x, y, z, c=colors)
        make_output_dir(self.xacro_handler.log_path / self.folder_name )
        pickle.dump(fig, open(self.xacro_handler.log_path / self.folder_name / f'Configuration{self.index}.pkl', 'wb'))
        if vis:
            plt.show()
        

    def save_to_csv(self, data):
        '''
        save the results to a csv file
        '''
        # Save to CSV        
        if isinstance(self.logger.handlers[0], logging.FileHandler):
            log_path = os.path.relpath(os.path.dirname(self.logger.handlers[0].baseFilename))
        else:
            log_path = "scripts/logs/design/evaluation"
            
        folder_name = Path(log_path) / 'ManipulabilityPoints'
        os.makedirs(folder_name, exist_ok=True)
        file_name = folder_name / f'manipulability_points_{self.index}.csv'
        
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Position", "Avg Manipulability"])  
            writer.writerows(data)



class ManipulabilityAnalysis(MinimalManipulabilityAnalysis):
    def __init__(self, 
                 args: DesignArguments, 
                 xacro_handler: XacroHandler, 
                 logger: logging.Logger, 
                 wandb_config:dict, 
                 main_path:str, 
                 env) -> None:
        super().__init__(logger, wandb_config, main_path, env)
        self.xacro_handler = xacro_handler
        self.evaluation_res = []
        self.training = None

    def evaluate_design(self, config: dict, budget: float) -> Tuple[float, float]:
        '''
        main method called to evaluate a configuration
        '''
        # write design to xacro file
        self.xacro_handler.write_config_to_file(config=config)
        self.relaunch_ros()
        self.env.reload_robot_description()
        time.sleep(3)

        # calculate manipulability for node
        np.random.seed(self.wandb_config.seed)
        avg_manipulability = self.get_manipulability_avg(1)
        self.plot_manipulability_3d(vis=False)
        self.logger.info(f"configuration: {config}")
        self.logger.info(f"avg manipulability: {avg_manipulability}")
        loss = self.calculate_loss(avg_manipulability)

        self.evaluation_res.append(loss)
        self.index += 1
        return loss, avg_manipulability

    def calculate_loss(self, manipulability: float) -> float:
        '''
        I calculated a manipulability value and i have to parse it to a loss to move on from here with the other algorithm
        loss is between 0 and n(evaluation_tasks)
        As it is just a guess i wont start under 50-70% of the max-loss (n(evaluation_tasks))
        '''

        n_evaluation_tasks = 5
        # best value is 60% of the loss
        start_percentage = 0.5
        lower_bound = n_evaluation_tasks * start_percentage
        # range inbetween the manipulability changes the loss
        manipulability_range = n_evaluation_tasks - lower_bound
        # borders of the manipulability that is computed
        min_border = 0.0
        max_border =  15.0
        scale_factor = np.interp(manipulability, [min_border, max_border], [0, 1])
        loss = lower_bound + manipulability_range * (1- scale_factor)
        return loss
        
    # def save_to_csv(self, data):
    #     '''
    #     save the results to a csv file
    #     '''
    #     # Save to CSV
    #     folder_name = self.xacro_handler.log_path / 'ManipulabilityPoints'
    #     os.makedirs(folder_name, exist_ok=True)
    #     file_name = folder_name / f'manipulability_points_{self.index}.csv'
        
    #     with open(file_name, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["Position", "Avg Manipulability"])  
    #         writer.writerows(data)
