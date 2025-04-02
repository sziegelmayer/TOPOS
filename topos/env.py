import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_default_nnunet_path(var: str, fallback_subdir: str) -> str:
    """
    Return the default path for nnUNet env variables if not set by user.
    Uses ~/.nnunet/ by default.
    """
    if var in os.environ:
        return os.environ[var]
    
    default_path = str(Path.home() / ".nnunet" / fallback_subdir)
    os.environ[var] = default_path
    logger.info(f"{var} not set. Using default: {default_path}")
    os.makedirs(default_path, exist_ok=True)
    return default_path


def setup_nnunet_env():
    """
    Ensures that nnUNet environment variables are defined.
    Assigns defaults if they are not already set by the user.
    """
    get_default_nnunet_path("nnUNet_raw", "nnUNet_raw")
    get_default_nnunet_path("nnUNet_preprocessed", "nnUNet_preprocessed")
    get_default_nnunet_path("nnUNet_results", "nnUNet_results")
