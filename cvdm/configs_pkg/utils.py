from typing import Any, Dict, Optional

import yaml

from cvdm.configs_pkg.data_config import DataConfig
from cvdm.configs_pkg.eval_config import EvalConfig
from cvdm.configs_pkg.model_config import ModelConfig
from cvdm.configs_pkg.neptune_config import NeptuneConfig
from cvdm.configs_pkg.training_config import TrainingConfig


def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        config_data: Dict[str, Any] = yaml.safe_load(file)
    return config_data


def create_model_config(config_data: Dict[str, Any]) -> ModelConfig:
    raw_model = dict(config_data["model"])
    allowed_keys = {
        "noise_model_type",
        "alpha",
        "snr_expansion_n",
        "load_weights",
        "load_mu_weights",
        "zmd",
        "diff_inp",
    }
    model_data = {key: value for key, value in raw_model.items() if key in allowed_keys}

    if not model_data:
        model_data = {
            "noise_model_type": raw_model.get("name", "unet"),
            "alpha": raw_model.get("alpha", 0.001),
            "snr_expansion_n": raw_model.get("snr_expansion_n", 1),
            "load_weights": raw_model.get("load_weights", None),
            "load_mu_weights": raw_model.get("load_mu_weights", None),
            "zmd": raw_model.get("zmd", False),
            "diff_inp": raw_model.get("diff_inp", False),
        }
    else:
        model_data.setdefault("noise_model_type", raw_model.get("name", "unet"))
        model_data.setdefault("alpha", raw_model.get("alpha", 0.001))
        model_data.setdefault("snr_expansion_n", raw_model.get("snr_expansion_n", 1))
        model_data.setdefault("load_weights", raw_model.get("load_weights", None))
        model_data.setdefault("load_mu_weights", raw_model.get("load_mu_weights", None))
        model_data.setdefault("zmd", raw_model.get("zmd", False))
        model_data.setdefault("diff_inp", raw_model.get("diff_inp", False))

    return ModelConfig(**model_data)


def create_training_config(config_data: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**config_data["training"])


def create_data_config(config_data: Dict[str, Any]) -> DataConfig:
    data = dict(config_data["data"])
    if "dataset_path" not in data:
        datasets = config_data.get("datasets")
        base_path = data.get("dataset_base_path")
        if datasets and base_path:
            data["dataset_path"] = f"{base_path}/{datasets[0]}"
        else:
            raise KeyError(
                "data.dataset_path is required unless datasets and data.dataset_base_path are provided."
            )
    return DataConfig(**data)


def create_eval_config(config_data: Dict[str, Any]) -> EvalConfig:
    eval_data = dict(config_data["eval"])
    if "output_path" not in eval_data:
        top_level_output = config_data.get("output_path") or config_data.get("output_dir")
        if top_level_output:
            eval_data["output_path"] = top_level_output
        else:
            datasets = config_data.get("datasets")
            base_path = eval_data.get("output_base_path")
            if datasets and base_path:
                eval_data["output_path"] = f"{base_path}/{datasets[0]}"
            else:
                raise KeyError(
                    "eval.output_path is required unless datasets and eval.output_base_path are provided."
                )
    return EvalConfig(**eval_data)


def create_neptune_config(config_data: Dict[str, Any]) -> Optional[NeptuneConfig]:
    if "neptune" in config_data:
        return NeptuneConfig(**config_data["neptune"])
    else:
        return None
