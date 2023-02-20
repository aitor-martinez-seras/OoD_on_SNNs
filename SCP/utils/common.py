import logging
import sys
from pathlib import Path

import tomli


def find_idx_of_class(searched_class, labels, number_of_searched_samples=0, initial_pos=0):
    """
    Function that outputs a list with all the indices (positions) of the searched_class in the labels.
    This way, a List is obtained which can be used to extract the samples of a specific class from
    a vector which contains these samples in same order as the labels vector.
    Does not throw an error if the the number of searched samples is not reached

    Parameters
    ----------
    searched_class: array-like of shape (n_samples,),
        The class where are searching for in the labels vector

    labels: array like
        The predictions or labels where the specific class must be found

    number_of_searched_samples: integer, default 0
        Number of occurrences wanted to be found in the labels vector. If 0, search for all occurrences

    initial_pos: integer, default 0
        If provided, the search in the labels vector is done starting from initial_pos

    Returns
    ----------
    indices: List
        The indices of the positions where the searched class is found in the labels vector
    """
    indices = []

    # Case of searching for all the array
    if number_of_searched_samples == 0:
        for index, label in enumerate(labels[initial_pos:]):
            if label == searched_class:
                indices.append(initial_pos + index)

    # Case of searching only n number of indices
    else:
        i = 0
        for index, label in enumerate(labels[initial_pos:]):
            if i >= number_of_searched_samples:
                break
            if label == searched_class:
                indices.append(initial_pos + index)
                i += 1

    return indices


def load_config(conf_name):
    """
    Loads the configuration file in .toml format
    """
    with open(Path(rf"config/{conf_name}.toml"), mode="rb") as fp:
        conf = tomli.load(fp)
    return conf


def get_batch_size(config: dict, in_dataset: str, logger: logging.Logger):
    try:  # If the key exists, it means a specific batch size is defined for the dataset
        batch_size = config["hyperparameters"][in_dataset]
        logger.warning(f"Using custom batch_size = {batch_size} for {in_dataset}")
    except KeyError:
        batch_size = config["hyperparameters"]["batch_size"]
    return batch_size


def my_custom_logger(logger_name, logs_pth, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(logs_pth / f"{logger_name}.log", mode='w')

    # Set handler levels
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatter and assign to handlers
    format_string = "%(asctime)s — %(levelname)s — %(message)s"
    log_format = logging.Formatter(format_string)
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_str_for_ood_method_results(method_name, auroc, aupr, fpr95, fpr80, temp='No temp'):
    string = f""" Results for {method_name}:
    \tAUROC:\t{auroc}
    \tAUPR:\t{aupr}
    \tFPR95:\t{fpr95}
    \tFPR80:\t{fpr80}
    \tTemp:\t{temp}"""
    return string
