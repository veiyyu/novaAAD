from pathlib import Path


def locate_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return current.parent.parent


PROJECT_ROOT = locate_project_root()
DATA_DIR = PROJECT_ROOT / "datasets"

# modify this path to point to the dataset
DATASET = DATA_DIR / "COG-BCI/sub-01/ses-S1/eeg/PVT.set"
# DATASET = DATA_DIR / "CAP-POS/DDE-OP-3345rev02 electrode positions for CA-208.elc"


# Constants needed for labeling and filtering
SAMPLE_RATE = 128  # Hz
WINDOW_SIZE = 2000  # ms
SAMPLE_SIZE = SAMPLE_RATE * WINDOW_SIZE // 1000  # samples per window
