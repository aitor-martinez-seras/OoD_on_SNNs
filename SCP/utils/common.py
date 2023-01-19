import logging
from pathlib import Path

import tomli


def find_idx_of_class(searched_class, labels, n=0, initial_pos=0):
    """
    Function that outputs a list with all the indices of the searched_class in the labelsList
    If n is provided, only the first n coincidences are outputted
    If initial_pos is provided, the search starts by this position.
    searched_class, n, initial_pos -> integer
    labels -> array
    """
    indices = []
    if n == 0:
        # Case of searching for all the array
        for index, labels in enumerate(labels[initial_pos:]):
            if labels == searched_class:
                indices.append(initial_pos + index)
    else:
        # Case of searching only n number of indices
        i = 0
        for index, labels in enumerate(labels[initial_pos:]):
            if i >= n:
                break
            if labels == searched_class:
                indices.append(initial_pos + index)
                i += 1
    return indices


def load_config(conf_name):
    with open(Path(rf"config/{conf_name}.toml"), mode="rb") as fp:
        conf = tomli.load(fp)
    return conf


def load_paths_config() -> dict:
    print(f'Loading path from paths.toml')
    return load_config('paths')


def get_batch_size(config: dict, in_dataset: str, logger: logging.Logger):
    try:  # If the key exists, it means a specific batch size is defined for the dataset
        batch_size = config["hyperparameters"][in_dataset]
        logger.warning(f"Using custom batch_size = {batch_size} for {in_dataset}")
    except KeyError:
        batch_size = config["hyperparameters"]["batch_size"]
    return batch_size




