import os
import sys

# Get the correct project root (3 levels up from this file: src/utils/config.py → project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

for d in [DATA_DIR, RESULTS_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
PCA_VARIANCE = 0.95
