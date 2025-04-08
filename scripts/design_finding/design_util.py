import os
from typing import List, Any
from pathlib import Path
import yaml

def load_yaml_file(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def merge_configs(default_config, override_config):
    """
    Merge two dictionaries, with values from the override dictionary
    overriding values from the default dictionary.

    Parameters:
    - default_dict (dict): The dictionary with default values.
    - override_dict (dict): The dictionary with values to override.

    Returns:
    - dict: The merged dictionary.
    """
    for config, _ in override_config.items():
        for key, value in override_config[config].items():
            # Override the value if the key exists, otherwise add a new key-value pair
            default_config[config][key] = value  

    return default_config

def get_subdirectories(directory: str) -> List[str]:
    '''
    Get a list of subdirectories within a given directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of subdirectory names.

    '''
    # Get a list of all items in the directory
    items = os.listdir(directory)
    
    # Filter out non-directory items
    subdirectories = [d for d in items if os.path.isdir(os.path.join(directory, d))]
    
    return subdirectories


def get_subfolder(folder_path: Path) -> Path:
    '''
    Get a list of subdirs in the current folder
    
    '''
    subdirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # Check if there is exactly one subfolder
    if len(subdirs) >= 1:
        # Navigate into the first subfolder
        new_folder = subdirs[0]
        return get_subfolder(folder_path+"/"+new_folder)
    else:
        return folder_path


def make_output_dir(output_path: Path) -> None:
    '''
    Make a directory if it does not exist.
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def find_index_by_config_id(lst: list, target_id: Any):
    '''
    Find the index of a config_id in a list of dictionaries.
    '''
    for i, d in enumerate(lst):
        if d.config_id == target_id:
            return i
    return -1  # Return -1 if the target config_id is not found