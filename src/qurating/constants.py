from pathlib import Path

ROOT = Path(__file__).parents[2]

DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True, parents=True)

TEMPLATES_DIR = Path(__file__).parent / "prompting" / "templates"

LOG_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"