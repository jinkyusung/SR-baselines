import argparse
import json
from pathlib import Path
from typing import Dict
import yaml



def parse_config(dataset: str, model: str) -> argparse.Namespace:
    """Loads the YAML configuration for the specified dataset and model.

    This function constructs the file path based on the provided dataset and model
    names, loads the YAML file, and converts the dictionary to an argparse.Namespace
    object using json serialization. This enables nested dot notation access 
    (e.g., config.wandb.entity).

    Args:
        dataset (str): The name of the dataset.
        model (str): The name of the model architecture.

    Returns:
        argparse.Namespace: The configuration parameters as a Namespace object.

    Raises:
        FileNotFoundError: If the configuration file does not exist at the expected path.
    """
    config_path = Path(__file__).resolve().parent.parent / 'config' / model / f'{dataset}.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return json.loads(
            json.dumps(config_dict), 
            object_hook=lambda d: argparse.Namespace(**d)
        )
    else:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")



def parse_argument() -> argparse.Namespace:
    """Parses command-line arguments for experiment configuration.

    This function initializes an ArgumentParser to accept required arguments
    for the dataset and model architecture. It parses the arguments provided
    in the command line and returns them as a namespace object.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
            The object has the following attributes:
            - dataset (str): The name of the dataset to be used.
            - model (str): The name of the model architecture.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
    )
    args = parser.parse_args()
    return args



def load_config() -> Dict[str, any]:
    _args = parse_argument()
    config = parse_config(dataset=_args.dataset, model=_args.model)
    vars(config).update(vars(_args))
    return config
