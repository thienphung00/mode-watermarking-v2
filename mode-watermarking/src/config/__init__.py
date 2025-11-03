# Config package exports

from .mode_config import (
	OperationMode,
	ModeDefaults,
	EmbedSettings,
	TrainSettings,
	EvalSettings,
	LoggingSettings,
	ModeConfig,
	ConfigLoader as ModeConfigLoader,  # Rename to avoid conflict
	resolve_mode,
	merge_dicts,
)

from .config_loader import ConfigLoader  # New YAML config loader
