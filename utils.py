"""
Shared utility functions used across every stage of the pipeline.
Keeping this at the repo root lets each numbered folder's script import
it with a single `sys.path` append, without turning the project into a
formal installable package.
"""

import os
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str = None) -> dict:
    """Load the central YAML config. Defaults to config/config.yaml at repo root."""
    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Resolve a path from config.yaml (given relative to repo root) to an absolute path."""
    return os.path.join(REPO_ROOT, relative_path)


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def add_repo_root_to_path():
    """Allow `import utils` to work when a script is run from within a numbered folder."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
