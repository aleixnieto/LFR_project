"""CLI and full synthetic smoke tests for the SemiCLIP workflow."""

import argparse
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PROJECT_ROOT / "open-clip"
sys.path.insert(0, str(OPEN_CLIP_DIR))

import main_semiclip  # noqa: E402
from semiclip_concepts import ConceptMiningResult  # noqa: E402
from semiclip_config import load_semiclip_config  # noqa: E402
from semiclip_model import encode_text_embeddings  # noqa: E402


DEFAULT_CONFIG = OPEN_CLIP_DIR / "configs" / "semiclip" / "semiclip_rsicd.yaml"
CONCEPTS = ("airport", "road", "river", "field")


class TinyTokenizer:
    context_length = 18

    def __init__(self) -> None:
        self.concept_ids = {name: 20 + index for index, name in enumerate(CONCEPTS)}

    def __call__(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            if text == "":
                content = []
            elif text == "a photo includes":
                content = [1, 2, 3]
            elif text in self.concept_ids:
                content = [self.concept_ids[text]]
            else:
                content = [30]
            row = [10, *content, 99]
            row.extend([0] * (self.context_length - len(row)))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)


class MixingTransformer(nn.Module):
    batch_first = True

    def get_cast_dtype(self):
        return torch.float32

    def forward(self, values, attn_mask=None):
        return values + values.mean(dim=1, keepdim=True)


class TinyOpenCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(4, 3, bias=False)
        self.token_embedding = nn.Embedding(128, 5)
        self.positional_embedding = nn.Parameter(torch.zeros(18, 5))
        self.transformer = MixingTransformer()
        self.attn_mask = None
        self.text_pool_type = "argmax"
        self.text_eos_id = None
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.randn(5, 3))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        flattened = images.reshape(images.shape[0], -1)
        return self.image_projection(flattened[:, :4])

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return encode_text_embeddings(self, tokens, self.token_embedding(tokens))


def image_transform(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image.resize((2, 2)))


def metadata_record(filename: str, split: str, index: int) -> dict:
    return {
        "filename": filename,
        "split": split,
        "sentences": [
            {"tokens": ["caption", str(index), "first"]},
            {"tokens": ["caption", str(index), "second"]},
        ],
    }


class MainSemiCLIPTests(unittest.TestCase):
    def test_parser_exposes_stage_and_bounded_smoke_controls(self) -> None:
        args = main_semiclip.parse_args(
            [
                "--stage",
                "2",
                "--device",
                "cpu",
                "--stage2-epochs",
                "3",
                "--max-stage2-train-batches",
                "2",
                "--max-validation-batches",
                "1",
                "--max-unlabeled-samples",
                "16",
            ]
        )

        self.assertEqual(args.stage, "2")
        self.assertEqual(args.stage2_epochs, 3)
        self.assertEqual(args.max_stage2_train_batches, 2)
        self.assertEqual(args.max_validation_batches, 1)
        self.assertEqual(args.max_unlabeled_samples, 16)

        with self.assertRaises(SystemExit):
            main_semiclip.parse_args(["--max-stage1-train-batches", "0"])
        with self.assertRaises(SystemExit):
            main_semiclip.parse_args(
                ["--stage", "1", "--resume-stage2", "checkpoint.pt"]
            )

    def test_concept_artifact_roundtrip_rejects_split_drift(self) -> None:
        training = [metadata_record(f"train_{index}.jpg", "val", index) for index in range(4)]
        validation = [metadata_record("validation.jpg", "val", 5)]
        result = ConceptMiningResult(
            concepts=CONCEPTS,
            counts={name: 2 for name in CONCEPTS},
            labels_by_filename={
                record["filename"]: (index % len(CONCEPTS),)
                for index, record in enumerate(training)
            },
            source_filenames=tuple(record["filename"] for record in training),
        )
        config = load_semiclip_config(DEFAULT_CONFIG)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "concepts.json"
            main_semiclip.write_workflow_concepts(
                path, result, validation, config
            )
            restored = main_semiclip.load_workflow_concepts(
                path, training, validation
            )
            self.assertEqual(restored, result)

            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["labeled_validation_filenames"] = ["different.jpg"]
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation filenames"):
                main_semiclip.load_workflow_concepts(path, training, validation)

    def test_all_stage_cpu_smoke_creates_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images"
            image_dir.mkdir()
            records = []
            for index in range(6):
                filename = f"val_{index}.jpg"
                Image.new("RGB", (24, 24), color=(20 + index, 40, 60)).save(
                    image_dir / filename
                )
                records.append(metadata_record(filename, "val", index))
            for index in range(4):
                filename = f"train_{index}.jpg"
                Image.new("RGB", (24, 24), color=(60, 20 + index, 40)).save(
                    image_dir / filename
                )
                records.append(
                    {
                        "filename": filename,
                        "split": "train",
                        "sentences": [{"malformed": "must never be used"}],
                    }
                )
            filename = "test.jpg"
            Image.new("RGB", (24, 24), color="white").save(image_dir / filename)
            records.append(metadata_record(filename, "test", 99))
            json_path = root / "dataset.json"
            json_path.write_text(json.dumps({"images": records}), encoding="utf-8")
            output_dir = root / "output"
            config = replace(
                load_semiclip_config(DEFAULT_CONFIG),
                image_dir=image_dir,
                json_path=json_path,
                output_dir=output_dir,
                total_batch_size=4,
                num_workers=0,
                stage1_epochs=1,
                stage2_epochs=1,
                warmup_steps=0,
                learning_rate=1e-3,
                concept_min_frequency=1,
                concept_max_frequency_ratio=1.0,
                selection_ratio=0.5,
            )
            args = argparse.Namespace(
                stage="all",
                config=DEFAULT_CONFIG,
                device="cpu",
                output_dir=None,
                resume_stage2=None,
                stage1_epochs=None,
                stage2_epochs=None,
                max_stage1_train_batches=1,
                max_stage2_train_batches=1,
                max_validation_batches=1,
                max_unlabeled_samples=4,
            )
            tokenizer = TinyTokenizer()
            model = TinyOpenCLIP()

            def fake_mining(training_records, **_kwargs):
                filenames = tuple(record["filename"] for record in training_records)
                return ConceptMiningResult(
                    concepts=CONCEPTS,
                    counts={name: len(filenames) for name in CONCEPTS},
                    labels_by_filename={
                        filename: (index % len(CONCEPTS),)
                        for index, filename in enumerate(filenames)
                    },
                    source_filenames=filenames,
                )

            with (
                patch.object(main_semiclip, "require_nltk_resources"),
                patch.object(
                    main_semiclip,
                    "mine_semantic_concepts",
                    side_effect=fake_mining,
                ),
                patch.object(
                    main_semiclip,
                    "load_openclip_components",
                    return_value=(
                        model,
                        tokenizer,
                        image_transform,
                        image_transform,
                    ),
                ),
            ):
                main_semiclip.run_semiclip(args, config)

            expected = {
                "resolved_config.json",
                "concepts.json",
                "stage1_best.pt",
                "stage1_final.pt",
                "stage1_unlabeled_logits.pt",
                "stage1_metrics.json",
                "stage2_last.pt",
                "stage2_final.pt",
                "stage2_metrics.json",
            }
            self.assertTrue(expected.issubset(path.name for path in output_dir.iterdir()))
            stage2 = torch.load(
                output_dir / "stage2_final.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(stage2["checkpoint_kind"], "stage2")
            self.assertEqual(stage2["artifact_role"], "final")
            self.assertEqual(stage2["concepts"], list(CONCEPTS))
            cache = torch.load(
                output_dir / "stage1_unlabeled_logits.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(set(cache["filenames"]), {f"train_{i}.jpg" for i in range(4)})


if __name__ == "__main__":
    unittest.main()
