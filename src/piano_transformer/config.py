from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    base_path: Path
    data_raw_path: Path
    transformer_name: str
    tokenizer_name: str
    dataset_name: str
    model_version: str
    model_name: str
    model_base_path: Path
    data_processed_path: Path
    model_path: Path
    runs_path: Path
    output_path: Path
    seed: int


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    base_path = Path(d["data"]["base_path"])
    data_raw_path = base_path / "data"
    transformer_name = d["model"]["transformer_name"]
    tokenizer_name = d["model"]["tokenizer_name"]
    dataset_name = d["model"]["dataset_name"]
    model_version = str(d["model"]["version"])
    model_name = f"{transformer_name}_{tokenizer_name}_{dataset_name}_v{model_version}"
    model_base_path = base_path / "models" / model_name
    data_processed_path = model_base_path / "data_processed"
    model_path = model_base_path / "model"
    runs_path = model_base_path / "runs"
    output_path = model_base_path / "output"
    seed = d["seed"]

    return Config(
        base_path=base_path,
        data_raw_path=data_raw_path,
        transformer_name=transformer_name,
        tokenizer_name=tokenizer_name,
        dataset_name=dataset_name,
        model_version=model_version,
        model_name=model_name,
        model_base_path=model_base_path,
        data_processed_path=data_processed_path,
        model_path=model_path,
        runs_path=runs_path,
        output_path=output_path,
        seed=seed,
    )
