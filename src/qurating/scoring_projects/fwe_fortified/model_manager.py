from pathlib import Path
import subprocess

from datasets import Dataset
from transformers import AutoConfig

from qurating.constants import RESULTS_DIR, ROOT, DATA_DIR


class QuratingModelManager:
    def __init__(self):
        self._keep_files = [
            "*.json",
            "*.safetensors",
            "*.jinja",
            "*.txt",
        ]
        self._s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
        self._checkpoints_base = DATA_DIR / "checkpoints-preferences"
        self._models_base = DATA_DIR / "qurating_models"
        self._s3_folder = (
            "s3://qurating-checkpoints-183631302286-eu-west-2-an/qurating_models/"
        )
        self._models_dataset_files = {
            "core_ed": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-500000_seed-72353534"
                / "ours_v2"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
            "fl_student": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-primary-5-pedagogical-5-500000_seed-274634520"
                / "FLN_student-facing"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
            "fl_teacher": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-pedagogical-5-500000_seed-274634520"
                / "FLN_teacher-facing"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
        }

        self.labels = {
            model_type: self._load_labels(dataset_file)
            for model_type, dataset_file in self._models_dataset_files.items()
        }

        self._models_checkpoints = {
            "qwen-3-4b": {
                "core_ed": str(
                    self._checkpoints_base
                    / "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs"
                    / "checkpoint-352"
                ),
                "fl_student": str(
                    self._checkpoints_base
                    / "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-primary-5-pedagogical-5-FLN_student-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-340"
                ),
                "fl_teacher": str(
                    self._checkpoints_base
                    / "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-pedagogical-5-FLN_teacher-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-312"
                ),
            },
            "gemma-3-4b": {
                "core_ed": str(
                    self._checkpoints_base
                    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs"
                    / "checkpoint-352"
                ),
                "fl_student": str(
                    self._checkpoints_base
                    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-primary-5-pedagogical-5-FLN_student-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-340"
                ),
                "fl_teacher": str(
                    self._checkpoints_base
                    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-pedagogical-5-FLN_teacher-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-312"
                ),
            },
        }
        assert all(
            set(models) == set(self._models_dataset_files) for models in self._models_checkpoints.values()
        )

    def model_types(self, base_model):
        return (mod_type for mod_type in self._models_checkpoints[base_model])

    def upload_all(self, base_models: str | list[str] | None = None):
        if base_models is None:
            base_models = list(self._models_checkpoints)
        elif isinstance(base_models, str):
            base_models = [base_models]
        for base_model in base_models:
            for model_type in self._models_checkpoints[base_model]:
                self._upload_s3(base_model, model_type)

    def download_all(self, base_models: str | list[str] | None = None):
        if base_models is None:
            base_models = list(self._models_checkpoints)
        elif isinstance(base_models, str):
            base_models = [base_models]
        for base_model in base_models:
            for model_type in self._models_checkpoints[base_model]:
                self._download_s3(base_model, model_type)

    def model_folder(
        self,
        base_model: str,
        model_type: str,
        s3_path: bool = False,
        local_path: bool = False,
    ):
        if local_path and s3_path:
            raise ValueError("Either local_path or s3_path can be True, but not both")
        stem = f"qurating_{model_type}_{base_model}"
        if s3_path:
            return f"{self._s3_folder}{stem}"
        elif local_path:
            return f"{self._models_base / stem}"
        else:
            return stem

    @staticmethod
    def _load_dataset(dataset_file: str | Path):
        parquetf = Path(dataset_file).with_suffix(".parquet")
        if parquetf.exists():
            ds = Dataset.from_parquet(str(dataset_file))
        elif Path(dataset_file).is_dir():
            ds = Dataset.load_from_disk(dataset_file)
        else:
            raise ValueError(f"{dataset_file} doesn't exist or is not a valid format")
        if not isinstance(ds, Dataset):
            raise ValueError(f"dataset file must return a {type(Dataset)} object")

        return ds

    def _load_labels(self, dataset_file: str | Path):
        ds = self._load_dataset(dataset_file)
        labels = [
            "_".join(col.split("_")[:-1])
            for col in ds.column_names
            if col.endswith("_average")
        ]
        return labels

    def _fix_labels(self, base_model: str, model_type: str):
        model_path_checkpoint = self._models_checkpoints[base_model][model_type]
        labels = self.labels[model_type]
        config = AutoConfig.from_pretrained(model_path_checkpoint)
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in id2label.items()}
        if id2label != config.id2label or label2id != config.label2id:
            config.id2label = id2label
            config.label2id = label2id
            config.save_pretrained(model_path_checkpoint)

    def _upload_s3(self, base_model: str, model_type: str):
        self._fix_labels(base_model, model_type)
        model_path = self._models_checkpoints[base_model][model_type]

        for fl in self._keep_files:
            cmd = [self._s5cmd_path, "--log=error"]
            cmd += ["cp", "-s", "-u", "-sp"]
            cmd += [
                str(f"{model_path}/{fl}"),
                f"{self.model_folder(base_model, model_type, s3_path=True)}/",
            ]
            subprocess.run(cmd)

    def _download_s3(self, base_model: str, model_type: str):
        model_name = self.model_folder(base_model, model_type)

        cmd = [self._s5cmd_path, "--log=error"]
        cmd += ["cp", "-n", "-s", "-u", "-sp"]
        cmd += [
            f"{self._s3_folder}{model_name}/*",
            f"{self._models_base}/{model_name}",
        ]
        subprocess.run(cmd)
