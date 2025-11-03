"""
Typed configuration and loader for mode-watermarking.

Centralizes parsing, validation, and access to configuration for all stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class OperationMode(Enum):
	"""Global operational modes for watermarking."""

	NON_DISTORTIONARY = "non_distortionary"
	DISTORTIONARY = "distortionary"


@dataclass
class ModeDefaults:
	seed: int
	device: str
	precision: str  # "float32" | "float16" | "bfloat16"
	output_root: str
	cache_root: str


@dataclass
class EmbedSettings:
	sd_model_id: str
	inference_steps: int
	guidance_scale: float
	batch_size: int
	mode: OperationMode
	key_id: str
	seed: int


@dataclass
class TrainSettings:
	dataset_manifest_train: str
	dataset_manifest_val: str
	batch_size: int
	epochs: int
	learning_rate: float
	scheduler: str
	weight_decay: float
	num_workers: int
	checkpoint_dir: str


@dataclass
class EvalSettings:
	dataset_manifest_test: str
	batch_size: int
	detector_checkpoint: str
	metrics: List[str]  # ["psnr", "ssim", "lpips", "fid"]
	threshold: float
	output_dir: str


@dataclass
class LoggingSettings:
	backend: str
	project: Optional[str]
	run_name: Optional[str]
	log_dir: str
	tags: Optional[List[str]]


@dataclass
class ModeConfig:
	defaults: ModeDefaults
	embed: EmbedSettings
	train: TrainSettings
	eval: EvalSettings
	logging: LoggingSettings


class ConfigLoader:
	"""Loads and validates YAML configurations into typed dataclasses."""

	def __init__(self, base_dir: str):
		self.base_dir = Path(base_dir).resolve()

	def _load_yaml(self, yaml_path: str) -> Dict[str, Any]:
		cfg_path = (self.base_dir / yaml_path) if not yaml_path.startswith("/") else Path(yaml_path)
		with open(cfg_path, "r") as f:
			data = yaml.safe_load(f) or {}
		return data

	def load_embed(self, yaml_path: str) -> ModeConfig:
		data = self._load_yaml(yaml_path)
		defaults = self._parse_defaults(data.get("defaults", {}))
		logging = self._parse_logging(data.get("logging", {}))

		# sd_model section
		sd_model = data.get("sd_model", {})
		sd_model_id = sd_model.get("id", "runwayml/stable-diffusion-v1-5")

		# embed section
		e = data.get("embed", {})
		embed = EmbedSettings(
			sd_model_id=sd_model_id,
			inference_steps=int(e.get("inference_steps", 50)),
			guidance_scale=float(e.get("guidance_scale", 7.5)),
			batch_size=int(e.get("batch_size", 4)),
			mode=resolve_mode(e.get("mode", OperationMode.NON_DISTORTIONARY.value)),
			key_id=str(e.get("key_id", "default_key_001")),
			seed=int(e.get("seed", defaults.seed)),
		)

		# Train/Eval placeholders for cross-stage completeness
		train = TrainSettings(
			dataset_manifest_train=data.get("data", {}).get("train_manifest", "data/splits/train.json"),
			dataset_manifest_val=data.get("data", {}).get("val_manifest", "data/splits/val.json"),
			batch_size=int(data.get("training", {}).get("batch_size", 8)),
			epochs=int(data.get("training", {}).get("epochs", 100)),
			learning_rate=float(data.get("training", {}).get("learning_rate", 1e-3)),
			scheduler=str(data.get("training", {}).get("scheduler", "cosine")),
			weight_decay=float(data.get("training", {}).get("weight_decay", 1e-4)),
			num_workers=int(data.get("data", {}).get("num_workers", 4)),
			checkpoint_dir=str(data.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")),
		)

		eval_cfg = data.get("eval", {})
		metrics = data.get("metrics", {})
		eval_settings = EvalSettings(
			dataset_manifest_test=str(eval_cfg.get("test_manifest", "data/splits/test.json")),
			batch_size=int(eval_cfg.get("batch_size", 16)),
			detector_checkpoint=str(eval_cfg.get("detector_checkpoint", "outputs/checkpoints/best_model.ckpt")),
			metrics=list(metrics.get("list", ["psnr", "ssim", "lpips", "fid"])),
			threshold=float(eval_cfg.get("threshold", 0.5)),
			output_dir=str(metrics.get("save_dir", "outputs/metrics")),
		)

		return ModeConfig(
			defaults=defaults,
			embed=embed,
			train=train,
			eval=eval_settings,
			logging=logging,
		)

	def load_train(self, yaml_path: str) -> ModeConfig:
		data = self._load_yaml(yaml_path)
		defaults = self._parse_defaults(data.get("defaults", {}))
		logging = self._parse_logging(data.get("logging", {}))

		train = TrainSettings(
			dataset_manifest_train=str(data.get("data", {}).get("train_manifest", "data/splits/train.json")),
			dataset_manifest_val=str(data.get("data", {}).get("val_manifest", "data/splits/val.json")),
			batch_size=int(data.get("training", {}).get("batch_size", 8)),
			epochs=int(data.get("training", {}).get("epochs", 100)),
			learning_rate=float(data.get("training", {}).get("learning_rate", 1e-3)),
			scheduler=str(data.get("training", {}).get("scheduler", "cosine")),
			weight_decay=float(data.get("training", {}).get("weight_decay", 1e-4)),
			num_workers=int(data.get("data", {}).get("num_workers", 4)),
			checkpoint_dir=str(data.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")),
		)

		# carry over embed for downstream compatibility (minimal)
		embed = EmbedSettings(
			sd_model_id=str(data.get("sd_model", {}).get("id", "runwayml/stable-diffusion-v1-5")),
			inference_steps=int(data.get("embed", {}).get("inference_steps", 50)),
			guidance_scale=float(data.get("embed", {}).get("guidance_scale", 7.5)),
			batch_size=int(data.get("embed", {}).get("batch_size", 4)),
			mode=resolve_mode(data.get("embed", {}).get("mode", OperationMode.NON_DISTORTIONARY.value)),
			key_id=str(data.get("embed", {}).get("key_id", "default_key_001")),
			seed=int(data.get("embed", {}).get("seed", defaults.seed)),
		)

		eval_settings = EvalSettings(
			dataset_manifest_test=str(data.get("eval", {}).get("test_manifest", "data/splits/test.json")),
			batch_size=int(data.get("eval", {}).get("batch_size", 16)),
			detector_checkpoint=str(data.get("eval", {}).get("detector_checkpoint", "outputs/checkpoints/best_model.ckpt")),
			metrics=list(data.get("metrics", {}).get("list", ["psnr", "ssim", "lpips", "fid"])),
			threshold=float(data.get("eval", {}).get("threshold", 0.5)),
			output_dir=str(data.get("metrics", {}).get("save_dir", "outputs/metrics")),
		)

		return ModeConfig(
			defaults=defaults,
			embed=embed,
			train=train,
			eval=eval_settings,
			logging=logging,
		)

	def load_eval(self, yaml_path: str) -> ModeConfig:
		data = self._load_yaml(yaml_path)
		defaults = self._parse_defaults(data.get("defaults", {}))
		logging = self._parse_logging(data.get("logging", {}))

		eval_cfg = data.get("eval", {})
		metrics = data.get("metrics", {})
		eval_settings = EvalSettings(
			dataset_manifest_test=str(eval_cfg.get("test_manifest", "data/splits/test.json")),
			batch_size=int(eval_cfg.get("batch_size", 16)),
			detector_checkpoint=str(eval_cfg.get("detector_checkpoint", "outputs/checkpoints/best_model.ckpt")),
			metrics=list(metrics.get("list", ["psnr", "ssim", "lpips", "fid"])),
			threshold=float(eval_cfg.get("threshold", 0.5)),
			output_dir=str(metrics.get("save_dir", "outputs/metrics")),
		)

		# carry minimal train/embed for consumers that expect full ModeConfig
		train = TrainSettings(
			dataset_manifest_train=str(data.get("data", {}).get("train_manifest", "data/splits/train.json")),
			dataset_manifest_val=str(data.get("data", {}).get("val_manifest", "data/splits/val.json")),
			batch_size=int(data.get("training", {}).get("batch_size", 8)),
			epochs=int(data.get("training", {}).get("epochs", 100)),
			learning_rate=float(data.get("training", {}).get("learning_rate", 1e-3)),
			scheduler=str(data.get("training", {}).get("scheduler", "cosine")),
			weight_decay=float(data.get("training", {}).get("weight_decay", 1e-4)),
			num_workers=int(data.get("data", {}).get("num_workers", 4)),
			checkpoint_dir=str(data.get("training", {}).get("checkpoint_dir", "outputs/checkpoints")),
		)

		embed = EmbedSettings(
			sd_model_id=str(data.get("sd_model", {}).get("id", "runwayml/stable-diffusion-v1-5")),
			inference_steps=int(data.get("embed", {}).get("inference_steps", 50)),
			guidance_scale=float(data.get("embed", {}).get("guidance_scale", 7.5)),
			batch_size=int(data.get("embed", {}).get("batch_size", 4)),
			mode=resolve_mode(data.get("embed", {}).get("mode", OperationMode.NON_DISTORTIONARY.value)),
			key_id=str(data.get("embed", {}).get("key_id", "default_key_001")),
			seed=int(data.get("embed", {}).get("seed", defaults.seed)),
		)

		return ModeConfig(
			defaults=defaults,
			embed=embed,
			train=train,
			eval=eval_settings,
			logging=logging,
		)

	def to_dict(self, cfg: ModeConfig) -> Dict[str, Any]:
		"""Convert ModeConfig to serializable dict."""
		from dataclasses import asdict

		result = asdict(cfg)
		# Convert enums to values
		result["embed"]["mode"] = cfg.embed.mode.value
		return result

	def _parse_defaults(self, d: Dict[str, Any]) -> ModeDefaults:
		return ModeDefaults(
			seed=int(d.get("seed", 42)),
			device=str(d.get("device", "cuda")),
			precision=str(d.get("precision", "float32")),
			output_root=str(d.get("output_root", "outputs")),
			cache_root=str(d.get("cache_root", ".cache")),
		)

	def _parse_logging(self, d: Dict[str, Any]) -> LoggingSettings:
		return LoggingSettings(
			backend=str(d.get("backend", "console")),
			project=d.get("project"),
			run_name=d.get("run_name"),
			log_dir=str(d.get("log_dir", "outputs/experiments")),
			tags=list(d.get("tags", [])) if d.get("tags") is not None else None,
		)


def resolve_mode(value: "str | OperationMode") -> OperationMode:
	"""Resolve a mode from string or enum."""
	if isinstance(value, OperationMode):
		return value
	value_norm = str(value).strip().lower()
	if value_norm in {OperationMode.NON_DISTORTIONARY.value, "non-distortionary", "non_distortionary"}:
		return OperationMode.NON_DISTORTIONARY
	if value_norm in {OperationMode.DISTORTIONARY.value, "distortionary"}:
		return OperationMode.DISTORTIONARY
	raise ValueError(f"Unknown operation mode: {value}")


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	"""Shallow merge of two dictionaries with override precedence."""
	merged = dict(base)
	merged.update(override or {})
	return merged
