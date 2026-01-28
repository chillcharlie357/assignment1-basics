import logging
import os
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf

# Default configuration
DEFAULT_CONFIG = {
    "logging": {
        "name": "cs336",
        "level": "INFO",
        "save_dir": "logs",
        "filename": "app.log",
        "rich": {
            "show_path": True,
            "rich_tracebacks": True,
            "markup": True
        }
    }
}

def _get_val(obj, path, default):
    """Helper to get value from nested dict/object"""
    keys = path.split('.')
    val = obj
    try:
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, (DictConfig, object)):
                val = getattr(val, k)
            else:
                return default
        return val if val is not None else default
    except:
        return default

def setup_logging(cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    
    # Determine if we are passed the root config or just logging config
    logging_cfg = cfg
    if _get_val(cfg, "logging", None) is not None:
        logging_cfg = _get_val(cfg, "logging", None)

    name = _get_val(logging_cfg, "name", "cs336")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates/stale config
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Console Handler (Rich)
    level = _get_val(logging_cfg, "level", "INFO")
    rich_cfg = _get_val(logging_cfg, "rich", {})
    
    console_handler = RichHandler(
        level=level,
        show_path=_get_val(rich_cfg, "show_path", True),
        rich_tracebacks=_get_val(rich_cfg, "rich_tracebacks", True),
        markup=_get_val(rich_cfg, "markup", True)
    )

    logger.addHandler(console_handler)
    
    return logger

# Initialize with defaults
logger = setup_logging(DEFAULT_CONFIG)
