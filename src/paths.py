from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_CSV = PROJECT_ROOT / "splits" / "nih_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEV_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dev"
TEST_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test" # Gets imported in exactly one place (test entry point)
RESEARCH_DIR = PROJECT_ROOT / "research" # Committed home for the log, the analysis plan, the runs table, and the final test results (after test run)
