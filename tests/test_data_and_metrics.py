from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import analyze_scores
import torch
import train_esm_classifier as train
from metrics import binary_average_precision, binary_roc_auc


class DataAndMetricsTests(unittest.TestCase):
    def write_csv(self, content: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        handle.write(content)
        handle.close()
        return handle.name

    def test_load_variant_csv(self) -> None:
        path = self.write_csv(
            "wt_seq,mut_seq,label\n"
            "MKT,MRT,0\n"
            "MKT,MAT,1\n"
        )

        records = train.load_variant_csv(path, "wt_seq", "mut_seq", "label")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].label, 0)
        self.assertEqual(records[1].mut_seq, "MAT")

    def test_load_variant_csv_reads_optional_position(self) -> None:
        path = self.write_csv(
            "wt_seq,mut_seq,position,label\n"
            "MKT,MRT,2,0\n"
        )

        records = train.load_variant_csv(path, "wt_seq", "mut_seq", "label", "position")

        self.assertEqual(records[0].position, 2)

    def test_load_variant_csv_rejects_out_of_range_position(self) -> None:
        path = self.write_csv(
            "wt_seq,mut_seq,position,label\n"
            "MKT,MRT,4,0\n"
        )

        with self.assertRaisesRegex(ValueError, "outside the sequence length"):
            train.load_variant_csv(path, "wt_seq", "mut_seq", "label", "position")

    def test_load_variant_csv_rejects_missing_columns(self) -> None:
        path = self.write_csv("wt_seq,label\nMKT,1\n")

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            train.load_variant_csv(path, "wt_seq", "mut_seq", "label")

    def test_load_variant_csv_rejects_missing_requested_position_column(self) -> None:
        path = self.write_csv("wt_seq,mut_seq,label\nMKT,MRT,1\n")

        with self.assertRaisesRegex(ValueError, "missing required columns: position"):
            train.load_variant_csv(path, "wt_seq", "mut_seq", "label", "position")

    def test_load_variant_csv_rejects_non_binary_labels(self) -> None:
        path = self.write_csv("wt_seq,mut_seq,label\nMKT,MAT,2\n")

        with self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
            train.load_variant_csv(path, "wt_seq", "mut_seq", "label")

    def test_class_weights_are_data_driven(self) -> None:
        records = [
            train.VariantRecord("AAA", "AAT", 0),
            train.VariantRecord("AAA", "AAC", 0),
            train.VariantRecord("AAA", "AAG", 1),
        ]

        weights = train.compute_class_weights(records, "auto", torch.device("cpu"))

        self.assertAlmostEqual(float(weights[0]), 0.75)
        self.assertAlmostEqual(float(weights[1]), 1.5)

    def test_collate_variant_batch_dynamic_padding(self) -> None:
        batch = [
            {
                "wt_input_ids": torch.tensor([0, 5, 2]),
                "wt_attention_mask": torch.tensor([1, 1, 1]),
                "mut_input_ids": torch.tensor([0, 6, 2]),
                "mut_attention_mask": torch.tensor([1, 1, 1]),
                "label": torch.tensor(0),
                "position": torch.tensor(1),
            },
            {
                "wt_input_ids": torch.tensor([0, 5, 7, 2]),
                "wt_attention_mask": torch.tensor([1, 1, 1, 1]),
                "mut_input_ids": torch.tensor([0, 6, 8, 2]),
                "mut_attention_mask": torch.tensor([1, 1, 1, 1]),
                "label": torch.tensor(1),
                "position": torch.tensor(2),
            },
        ]

        collated = train.collate_variant_batch(batch, pad_token_id=99)

        self.assertEqual(collated["wt_input_ids"].shape, (2, 4))
        self.assertEqual(collated["wt_input_ids"][0, 3].item(), 99)
        self.assertEqual(collated["wt_attention_mask"][0, 3].item(), 0)
        self.assertEqual(collated["position"].tolist(), [1, 2])

    def test_pool_variant_embeddings_uses_position_when_available(self) -> None:
        hidden = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
            ]
        )

        pooled = train.pool_variant_embeddings(hidden, torch.tensor([2, -1]))

        self.assertEqual(pooled.tolist(), [[2.0, 2.0], [10.0, 10.0]])

    def test_feature_cache_round_trip(self) -> None:
        records = [
            train.VariantRecord("AAA", "AAT", 0, 3),
            train.VariantRecord("AAA", "AAC", 1, 3),
        ]
        args = SimpleNamespace(model_name="esm-test", max_len=8)
        metadata = train.feature_cache_metadata(records, args, "train", feature_dim=6)
        features = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        labels = torch.tensor([0, 1])
        cache_path = Path(tempfile.mkdtemp()) / "train.pt"

        train.save_feature_cache(cache_path, features, labels, metadata)
        dataset = train.load_cached_feature_dataset(cache_path, metadata)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.feature_dim, 6)
        self.assertEqual(dataset[1]["label"].item(), 1)
        self.assertTrue(torch.equal(dataset[0]["features"], features[0]))

    def test_feature_cache_rejects_metadata_mismatch(self) -> None:
        records = [train.VariantRecord("AAA", "AAT", 0, 3)]
        args = SimpleNamespace(model_name="esm-test", max_len=8)
        metadata = train.feature_cache_metadata(records, args, "train", feature_dim=3)
        cache_path = Path(tempfile.mkdtemp()) / "train.pt"

        train.save_feature_cache(cache_path, torch.ones(1, 3), torch.tensor([0]), metadata)
        stale_metadata = dict(metadata)
        stale_metadata["max_len"] = 4

        self.assertIsNone(train.try_load_cached_feature_dataset(cache_path, stale_metadata))

    def test_feature_dim_for_model_uses_config_hidden_size(self) -> None:
        with patch.object(train.AutoConfig, "from_pretrained", return_value=SimpleNamespace(hidden_size=128)):
            self.assertEqual(train.feature_dim_for_model("esm-test"), 384)

    def test_batch_outputs_and_labels_accepts_cached_features(self) -> None:
        model = train.FeatureOnlyClassifier(feature_dim=3)
        batch = {
            "features": torch.ones(2, 3),
            "label": torch.tensor([0, 1]),
        }

        outputs, labels = train.batch_outputs_and_labels(model, batch, torch.device("cpu"))

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_metrics_skip_auc_for_single_class_validation(self) -> None:
        metrics = train.evaluate_predictions([0, 0], [0.1, 0.2], [0, 0], loss=0.4)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["auprc"])

    def test_binary_metrics(self) -> None:
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]

        self.assertAlmostEqual(binary_roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(binary_average_precision(labels, scores), 1.0)

    def test_average_precision_groups_tied_scores(self) -> None:
        labels = [1, 0, 1]
        scores = [0.5, 0.5, 0.4]

        self.assertAlmostEqual(binary_average_precision(labels, scores), (0.5 + 2 / 3) / 2)

    def test_analyze_scores_patient_topk(self) -> None:
        path = self.write_csv(
            "Patient_ID,LABEL,SCORE_A,SCORE_B\n"
            "p1,0,0.1,0.9\n"
            "p1,1,0.8,0.2\n"
            "p2,0,0.2,0.1\n"
            "p3,1,0.3,0.4\n"
        )

        with contextlib.redirect_stdout(io.StringIO()):
            results = analyze_scores.analyze_scores(path, score_cols=["SCORE_A", "SCORE_B"])

        self.assertEqual(results["SCORE_A"]["Top-1 Accuracy"], 1.0)
        self.assertEqual(results["SCORE_A"]["Top-5 Hit"], 1.0)
        self.assertEqual(results["SCORE_B"]["Top-1 Accuracy"], 0.5)
        self.assertEqual(results["SCORE_B"]["Top-5 Hit"], 1.0)

    def test_analyze_scores_rejects_nan_scores(self) -> None:
        path = self.write_csv(
            "Patient_ID,LABEL,SCORE_A\n"
            "p1,1,nan\n"
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            analyze_scores.load_score_rows(path, "Patient_ID", "LABEL", ["SCORE_A"])

    def test_patient_hit_at_k_is_tie_inclusive(self) -> None:
        group = [
            {"label": 0, "scores": {"SCORE_A": 1.0}},
            {"label": 1, "scores": {"SCORE_A": 1.0}},
            {"label": 0, "scores": {"SCORE_A": 0.1}},
        ]

        self.assertTrue(analyze_scores.patient_hit_at_k(group, "SCORE_A", 1))


if __name__ == "__main__":
    unittest.main()
