from pathlib import Path
# Path for the datasets
root_path = Path(Path.cwd())
DATASETS_PATH = Path('SCP/datasets/')
PRETRAINED_WEIGHTS_PATH = Path('weights/pretrained')
FIGURES_PATH = Path('figures/')

# Constants for the correct visualization of the .csv in Spanish configuration
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','

# For the explainability part
WHICH_FPR = 80
