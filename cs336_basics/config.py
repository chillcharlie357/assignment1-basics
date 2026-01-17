import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class RichConfig(BaseModel):
    show_path: bool = True
    rich_tracebacks: bool = True
    markup: bool = True

class LoggingConfig(BaseModel):
    name: str = "cs336"
    level: str = "INFO"
    save_dir: str = "logs"
    filename: str = "app.log"
    rich: RichConfig = Field(default_factory=RichConfig)

class TokenizerDataConfig(BaseModel):
    input_path: str = "tests/fixtures/tinystories_sample_5M.txt"
    vocab_path: str = "data/vocab/tinystories_sample_5M_vocab.pkl"
    merges_path: str = "data/vocab/tinystories_sample_5M_merges.pkl"

class TokenizerTrainingConfig(BaseModel):
    vocab_size: int = 12800
    special_tokens: list[str] = ["<|endoftext|>"]

class TokenizerConfig(BaseModel):
    data: TokenizerDataConfig = Field(default_factory=TokenizerDataConfig)
    training: TokenizerTrainingConfig = Field(default_factory=TokenizerTrainingConfig)

class Config(BaseModel):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent

def load_config(config_path: str | Path = None) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file. If None, defaults to config/config.yaml in project root.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        return Config(**config_data)

# Global configuration object
try:
    config = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    # Fallback to defaults
    config = Config()
