# config.py
import yaml

def load_config(config_path='config.yaml'):
    """
    Load model configuration from a YAML file with error handling.
    
    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Parsed configuration data.
    """
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
