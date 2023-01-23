from pathlib import Path
import os

root_path = Path(os.path.dirname(__file__))
DATASETS_PATH = root_path / 'SCP/datasets'
WEIGHTS_PATH = root_path / 'weights'
PRETRAINED_WEIGHTS_PATH = root_path / 'weights/pretrained'
FIGURES_PATH = root_path / 'figures'

# Constants for the correct visualization of the .csv in Spanish configuration
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','

# For the explainability part
WHICH_FPR = 80
