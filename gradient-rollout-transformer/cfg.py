"""
Config for this project is stored in the `config.toml` file, please edit it accordingly
"""

import toml


def load_config(path="./config.toml") -> dict:
    """
    Load configuration from a TOML file.

    Parameters:
    - path (str): The path to the TOML configuration file.

    Returns:
    - dict: The loaded configuration data.
    """
    try:
        with open(path, "r") as config_file:
            config_data = toml.load(config_file)
        return config_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {path}: {e}")
