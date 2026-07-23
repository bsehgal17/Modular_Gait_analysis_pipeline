import os
import yaml
from typing import Optional
from pydantic import BaseModel
import logging

# pipeline‑level paths (dataset + ground‑truth)
from .pipeline_path import PipelinePathsConfig
from .models_config import ModelsConfig
from .processing_config import ProcessingConfig
from .filter_config import FilterConfig
from .noise_config import NoiseConfig
from .evaluation_config import EvaluationConfig
from .dataset_config import DatasetConfig  # <- new import
# <- confidence filtering import


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrackingConfig(BaseModel):
    overlap_threshold: float
    distance_threshold: float


class PipelineConfig(BaseModel):
    paths: PipelinePathsConfig
    models: Optional[ModelsConfig]
    processing: Optional[ProcessingConfig]
    filter: Optional[FilterConfig]
    noise: Optional[NoiseConfig]
    evaluation: Optional[EvaluationConfig]
    dataset: Optional[DatasetConfig]
    tracking: Optional[TrackingConfig]

    # ------------------------------------------------------------------
    # YAML (de)serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Pipeline config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            paths=PipelinePathsConfig(**raw_config.get("paths", {})),
            models=ModelsConfig(**raw_config["models"])
            if "models" in raw_config
            else None,
            processing=ProcessingConfig(**raw_config["processing"])
            if "processing" in raw_config
            else None,
            filter=FilterConfig(**raw_config["filter"])
            if "filter" in raw_config
            else None,
            noise=NoiseConfig(
                **raw_config["noise"]) if "noise" in raw_config else None,
            evaluation=EvaluationConfig(**raw_config["evaluation"])
            if "evaluation" in raw_config
            else None,
            dataset=DatasetConfig(**raw_config["dataset"])
            if "dataset" in raw_config
            else None,
            tracking=TrackingConfig(**raw_config["tracking"])
            if "tracking" in raw_config
            else None

        )

    def to_yaml(self, yaml_path: str):
        """Dump the current config to YAML."""
        cfg_dict = {}

        if self.paths:
            cfg_dict["paths"] = self.paths.model_dump()
        if self.models:
            cfg_dict["models"] = self.models.model_dump()
        if self.processing:
            cfg_dict["processing"] = self.processing.model_dump()
        if self.filter:
            cfg_dict["filter"] = self.filter.model_dump()
        if self.noise:
            cfg_dict["noise"] = self.noise.model_dump()
        if self.evaluation:
            cfg_dict["evaluation"] = self.evaluation.model_dump()
        if self.dataset:
            cfg_dict["dataset"] = self.dataset.model_dump()
        if self.tracking:
            cfg_dict["tracking"] = self.tracking.model_dump()

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, indent=4)


# ------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------
_pipeline_config: Optional[PipelineConfig] = None


def get_pipeline_config(config_file_path: Optional[str] = None) -> PipelineConfig:
    global _pipeline_config
    if _pipeline_config is None:
        if config_file_path:
            logger.info(f"Loading pipeline config from {config_file_path}")
            _pipeline_config = PipelineConfig.from_yaml(config_file_path)
        else:
            raise ValueError(
                "config_file_path must be provided for initial load of PipelineConfig."
            )
    return _pipeline_config
