#!/usr/bin/env python
"""Add classifier and FID error bars to existing benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process saved benchmark outputs with classifier/FID error bars.",
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="One or more benchmark result directories.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for classifier and FID intervals.",
    )
    parser.add_argument(
        "--fid-interval-method",
        choices=["paired_folds", "bootstrap"],
        default="paired_folds",
        help="Approximation used for FID uncertainty.",
    )
    parser.add_argument(
        "--fid-bootstrap-reps",
        type=int,
        default=200,
        help="Number of paired bootstrap replicates when --fid-interval-method=bootstrap.",
    )
    parser.add_argument(
        "--fid-fold-permutations",
        type=int,
        default=5,
        help="Number of random fold partitions when --fid-interval-method=paired_folds.",
    )
    parser.add_argument(
        "--fid-min-fold-size",
        type=int,
        default=10,
        help="Minimum fold size for paired fold FID intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Base RNG seed for FID bootstrap resampling.",
    )
    parser.add_argument(
        "--recompute-fid-point",
        action="store_true",
        help="Recompute the full FID point estimate from the saved features.",
    )
    parser.add_argument(
        "--summary-suffix",
        default="_with_error_bars",
        help="Suffix inserted before the output summary CSV extension.",
    )
    return parser.parse_args()


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    if len(matches) > 1:
        joined = ", ".join(str(path.name) for path in matches)
        raise RuntimeError(f"Expected one match for {pattern!r} under {directory}, found: {joined}")
    return matches[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "schedule_index": int(row["schedule_index"]),
                    "interval_start": float(row["interval_start"]),
                    "interval_end": float(row["interval_end"]),
                    "num_samples": int(row["num_samples"]),
                    "correct_count": int(row["correct_count"]),
                    "error_rate": float(row["error_rate"]),
                    "fid_to_global_conditioned": float(row["fid_to_global_conditioned"]),
                }
            )
    rows.sort(key=lambda row: row["schedule_index"])
    return rows


def torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, found {type(payload)!r}")
    return payload


def load_features(path: Path) -> np.ndarray:
    payload = torch_load_cpu(path)
    if "features" not in payload:
        raise KeyError(f"{path} does not contain a 'features' tensor")
    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"'features' in {path} is not a torch.Tensor")
    array = features.detach().cpu().numpy()
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature array in {path}, found shape {array.shape}")
    return np.asarray(array, dtype=np.float64)


def gram_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = lhs if right is None else np.asarray(right, dtype=np.float64)
    return lhs @ rhs.T


def percentile_bounds(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = 1.0 - ci_level
    low, high = np.percentile(
        np.asarray(values, dtype=np.float64),
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return float(low), float(high)


def wilson_interval(num_positive: int, num_total: int, ci_level: float) -> tuple[float, float]:
    if num_total < 1:
        raise ValueError("num_total must be positive")
    z = NormalDist().inv_cdf(0.5 + ci_level / 2.0)
    p_hat = num_positive / num_total
    z2 = z * z
    denom = 1.0 + z2 / num_total
    center = (p_hat + z2 / (2.0 * num_total)) / denom
    half = z * math.sqrt(
        (p_hat * (1.0 - p_hat) + z2 / (4.0 * num_total)) / num_total
    ) / denom
    return max(0.0, center - half), min(1.0, center + half)


def center_gram(gram: np.ndarray) -> np.ndarray:
    row_means = gram.mean(axis=1, keepdims=True)
    col_means = gram.mean(axis=0, keepdims=True)
    overall_mean = gram.mean()
    return gram - row_means - col_means + overall_mean


def fid_from_grams(
    reference_self_gram: np.ndarray,
    generated_self_gram: np.ndarray,
    cross_gram: np.ndarray,
) -> float:
    ref_gram = np.asarray(reference_self_gram, dtype=np.float64)
    gen_gram = np.asarray(generated_self_gram, dtype=np.float64)
    ref_gen_gram = np.asarray(cross_gram, dtype=np.float64)
    if ref_gram.ndim != 2 or gen_gram.ndim != 2 or ref_gen_gram.ndim != 2:
        raise ValueError("Expected 2D Gram matrices")

    diff_sq = float(ref_gram.mean() + gen_gram.mean() - 2.0 * ref_gen_gram.mean())

    trace_ref = 0.0
    if ref_gram.shape[0] >= 2:
        trace_ref = float(np.trace(center_gram(ref_gram)) / (ref_gram.shape[0] - 1))

    trace_gen = 0.0
    if gen_gram.shape[0] >= 2:
        trace_gen = float(np.trace(center_gram(gen_gram)) / (gen_gram.shape[0] - 1))

    cross = 0.0
    if ref_gram.shape[0] >= 2 and gen_gram.shape[0] >= 2:
        centered_cross = center_gram(ref_gen_gram)
        singular_values = np.linalg.svd(centered_cross, compute_uv=False)
        cross = float(
            singular_values.sum() / math.sqrt((ref_gram.shape[0] - 1) * (gen_gram.shape[0] - 1))
        )

    fid = diff_sq + trace_ref + trace_gen - 2.0 * cross
    if fid < 0.0 and fid > -1e-8:
        fid = 0.0
    return float(fid)


def bootstrap_fid_interval(
    reference_self_gram: np.ndarray,
    generated_self_gram: np.ndarray,
    cross_gram: np.ndarray,
    ci_level: float,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if reps < 1:
        raise ValueError("reps must be at least 1")

    ref_gram = np.asarray(reference_self_gram, dtype=np.float64)
    gen_gram = np.asarray(generated_self_gram, dtype=np.float64)
    ref_gen_gram = np.asarray(cross_gram, dtype=np.float64)
    rng = np.random.default_rng(seed)
    paired = ref_gram.shape[0] == gen_gram.shape[0]
    fids = np.empty(reps, dtype=np.float64)

    for rep in range(reps):
        if paired:
            indices = rng.integers(0, ref_gram.shape[0], size=ref_gram.shape[0])
            ref_boot = ref_gram[np.ix_(indices, indices)]
            gen_boot = gen_gram[np.ix_(indices, indices)]
            cross_boot = ref_gen_gram[np.ix_(indices, indices)]
        else:
            ref_indices = rng.integers(0, ref_gram.shape[0], size=ref_gram.shape[0])
            gen_indices = rng.integers(0, gen_gram.shape[0], size=gen_gram.shape[0])
            ref_boot = ref_gram[np.ix_(ref_indices, ref_indices)]
            gen_boot = gen_gram[np.ix_(gen_indices, gen_indices)]
            cross_boot = ref_gen_gram[np.ix_(ref_indices, gen_indices)]
        fids[rep] = fid_from_grams(ref_boot, gen_boot, cross_boot)

    low, high = percentile_bounds(fids, ci_level)
    return {
        "sample_values": fids,
        "bootstrap_mean": float(fids.mean()),
        "bootstrap_std": float(fids.std(ddof=1)) if reps > 1 else 0.0,
        "sample_median": float(np.median(fids)),
        "raw_ci_low": low,
        "raw_ci_high": high,
    }


def paired_fold_fid_interval(
    reference_self_gram: np.ndarray,
    generated_self_gram: np.ndarray,
    cross_gram: np.ndarray,
    ci_level: float,
    min_fold_size: int,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    if min_fold_size < 2:
        raise ValueError("min_fold_size must be at least 2")

    ref_gram = np.asarray(reference_self_gram, dtype=np.float64)
    gen_gram = np.asarray(generated_self_gram, dtype=np.float64)
    ref_gen_gram = np.asarray(cross_gram, dtype=np.float64)
    paired = ref_gram.shape[0] == gen_gram.shape[0]
    rng = np.random.default_rng(seed)
    fids: list[float] = []

    if paired:
        sample_count = ref_gram.shape[0]
        fold_count = max(2, min(10, sample_count // min_fold_size))
        fold_size = sample_count // fold_count
        used = fold_count * fold_size
        if fold_size < 2:
            raise ValueError("Not enough samples for paired fold FID interval")
        for _ in range(permutations):
            order = rng.permutation(sample_count)[:used]
            for fold_idx in range(fold_count):
                start = fold_idx * fold_size
                stop = start + fold_size
                indices = order[start:stop]
                ref_fold = ref_gram[np.ix_(indices, indices)]
                gen_fold = gen_gram[np.ix_(indices, indices)]
                cross_fold = ref_gen_gram[np.ix_(indices, indices)]
                fids.append(fid_from_grams(ref_fold, gen_fold, cross_fold))
        method = "paired_permutation_folds"
    else:
        ref_count = ref_gram.shape[0]
        gen_count = gen_gram.shape[0]
        fold_count = max(2, min(10, min(ref_count, gen_count) // min_fold_size))
        ref_fold_size = ref_count // fold_count
        gen_fold_size = gen_count // fold_count
        if ref_fold_size < 2 or gen_fold_size < 2:
            raise ValueError("Not enough samples for unpaired fold FID interval")
        ref_used = fold_count * ref_fold_size
        gen_used = fold_count * gen_fold_size
        for _ in range(permutations):
            ref_order = rng.permutation(ref_count)[:ref_used]
            gen_order = rng.permutation(gen_count)[:gen_used]
            for fold_idx in range(fold_count):
                ref_start = fold_idx * ref_fold_size
                ref_stop = ref_start + ref_fold_size
                gen_start = fold_idx * gen_fold_size
                gen_stop = gen_start + gen_fold_size
                ref_indices = ref_order[ref_start:ref_stop]
                gen_indices = gen_order[gen_start:gen_stop]
                ref_fold = ref_gram[np.ix_(ref_indices, ref_indices)]
                gen_fold = gen_gram[np.ix_(gen_indices, gen_indices)]
                cross_fold = ref_gen_gram[np.ix_(ref_indices, gen_indices)]
                fids.append(fid_from_grams(ref_fold, gen_fold, cross_fold))
        method = "unpaired_permutation_folds"

    fid_array = np.asarray(fids, dtype=np.float64)
    low, high = percentile_bounds(fid_array, ci_level)
    return {
        "sample_values": fid_array,
        "bootstrap_mean": float(fid_array.mean()),
        "bootstrap_std": float(fid_array.std(ddof=1)) if fid_array.size > 1 else 0.0,
        "sample_median": float(np.median(fid_array)),
        "raw_ci_low": low,
        "raw_ci_high": high,
        "num_estimates": int(fid_array.size),
        "method": method,
    }


def roundtrip_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No summary rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: roundtrip_jsonable(value) for key, value in row.items()})


def process_result_dir(
    result_dir: Path,
    ci_level: float,
    fid_interval_method: str,
    fid_bootstrap_reps: int,
    fid_fold_permutations: int,
    fid_min_fold_size: int,
    bootstrap_seed: int,
    summary_suffix: str,
    recompute_fid_point: bool,
) -> tuple[Path, Path]:
    metrics_dir = result_dir / "metrics"
    features_dir = result_dir / "features"
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Missing metrics directory: {metrics_dir}")
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Missing features directory: {features_dir}")

    summary_csv_path = find_one(metrics_dir, "*_summary.csv")
    metrics_json_path = find_one(metrics_dir, "*_metrics.json")
    metrics = load_json(metrics_json_path)
    rows = load_summary_rows(summary_csv_path)

    baseline_feature_path = features_dir / "baseline_global_conditioned_features.pt"
    baseline_features = load_features(baseline_feature_path)
    baseline_self_gram = gram_matrix(baseline_features)
    baseline = metrics.get("baseline", {})
    baseline_correct = int(baseline.get("correct_count", 0))
    baseline_num_samples = int(baseline.get("num_samples", baseline_features.shape[0]))
    baseline_errors = baseline_num_samples - baseline_correct
    baseline_ci_low, baseline_ci_high = wilson_interval(
        baseline_errors,
        baseline_num_samples,
        ci_level,
    )

    enriched_rows: list[dict[str, Any]] = []
    print(
        f"[postprocess] {result_dir} rows={len(rows)} "
        f"baseline_samples={baseline_features.shape[0]} "
        f"fid_interval_method={fid_interval_method}"
    )

    for row in rows:
        schedule_idx = row["schedule_index"]
        num_samples = row["num_samples"]
        error_count = num_samples - row["correct_count"]
        error_ci_low, error_ci_high = wilson_interval(error_count, num_samples, ci_level)

        feature_path = features_dir / f"schedule_{schedule_idx:02d}_features.pt"
        generated_features = load_features(feature_path)
        generated_self_gram = gram_matrix(generated_features)
        cross_gram = gram_matrix(baseline_features, generated_features)
        fid_point = row["fid_to_global_conditioned"]
        if recompute_fid_point:
            fid_point = fid_from_grams(baseline_self_gram, generated_self_gram, cross_gram)
        if fid_interval_method == "bootstrap":
            fid_stats = bootstrap_fid_interval(
                baseline_self_gram,
                generated_self_gram,
                cross_gram,
                ci_level=ci_level,
                reps=fid_bootstrap_reps,
                seed=bootstrap_seed + schedule_idx,
            )
            fid_stats["num_estimates"] = fid_bootstrap_reps
            fid_stats["method"] = (
                "paired_bootstrap_percentile"
                if baseline_features.shape[0] == generated_features.shape[0]
                else "unpaired_bootstrap_percentile"
            )
        else:
            fid_stats = paired_fold_fid_interval(
                baseline_self_gram,
                generated_self_gram,
                cross_gram,
                ci_level=ci_level,
                min_fold_size=fid_min_fold_size,
                permutations=fid_fold_permutations,
                seed=bootstrap_seed + schedule_idx,
            )
        raw_fid_values = np.asarray(fid_stats.pop("sample_values"), dtype=np.float64)
        raw_mean = float(fid_stats["bootstrap_mean"])
        raw_median = float(fid_stats["sample_median"])
        raw_ci_low = float(fid_stats["raw_ci_low"])
        raw_ci_high = float(fid_stats["raw_ci_high"])
        bias_estimate = raw_mean - fid_point
        centered_fid_values = raw_fid_values - bias_estimate
        centered_ci_low, centered_ci_high = percentile_bounds(centered_fid_values, ci_level)
        centered_method = f"{fid_stats['method']}_recentered_to_full_fid"

        enriched_row = dict(row)
        enriched_row.update(
            {
                "error_ci_low": error_ci_low,
                "error_ci_high": error_ci_high,
                "error_ci_half_width": max(
                    row["error_rate"] - error_ci_low,
                    error_ci_high - row["error_rate"],
                ),
                "fid_bootstrap_mean": fid_stats["bootstrap_mean"],
                "fid_bootstrap_std": fid_stats["bootstrap_std"],
                "fid_interval_median": raw_median,
                "fid_bias_estimate": bias_estimate,
                "fid_raw_ci_low": raw_ci_low,
                "fid_raw_ci_high": raw_ci_high,
                "fid_ci_low": centered_ci_low,
                "fid_ci_high": centered_ci_high,
                "fid_ci_half_width": max(
                    fid_point - centered_ci_low,
                    centered_ci_high - fid_point,
                ),
                "feature_count": int(generated_features.shape[0]),
                "fid_interval_method": centered_method,
                "fid_interval_raw_method": fid_stats["method"],
                "fid_num_estimates": fid_stats["num_estimates"],
            }
        )
        if recompute_fid_point:
            enriched_row["fid_recomputed"] = fid_point
            enriched_row["fid_recomputed_minus_reported"] = (
                fid_point - row["fid_to_global_conditioned"]
            )
        enriched_rows.append(enriched_row)
        print(
            f"  schedule={schedule_idx:02d} samples={generated_features.shape[0]} "
            f"error={row['error_rate']:.4f} "
            f"fid={row['fid_to_global_conditioned']:.4f} "
            f"raw_mean={raw_mean:.4f} "
            f"fid_ci=[{centered_ci_low:.4f}, {centered_ci_high:.4f}]"
        )

    summary_stem = summary_csv_path.stem
    summary_out_path = summary_csv_path.with_name(f"{summary_stem}{summary_suffix}.csv")
    base_prefix = summary_stem[:-8] if summary_stem.endswith("_summary") else summary_stem
    sidecar_path = metrics_dir / f"{base_prefix}_error_bars.json"

    write_summary_csv(enriched_rows, summary_out_path)
    sidecar = {
        "result_dir": str(result_dir),
        "source_summary_csv": str(summary_csv_path),
        "source_metrics_json": str(metrics_json_path),
        "output_summary_csv": str(summary_out_path),
        "classifier_error_method": {
            "metric": "error_rate",
            "distribution": "binomial",
            "interval": "wilson",
            "ci_level": ci_level,
        },
        "fid_method": {
            "metric": "fid_to_global_conditioned",
            "interval": enriched_rows[0]["fid_interval_method"],
            "raw_interval": enriched_rows[0]["fid_interval_raw_method"],
            "ci_level": ci_level,
            "bootstrap_reps": fid_bootstrap_reps,
            "fold_permutations": fid_fold_permutations,
            "min_fold_size": fid_min_fold_size,
            "bootstrap_seed": bootstrap_seed,
            "bias_correction": "translate interval samples so their mean matches the full-sample FID point estimate",
        },
        "baseline": {
            "num_samples": baseline_num_samples,
            "correct_count": baseline_correct,
            "error_rate": float(baseline.get("error_rate", baseline_errors / baseline_num_samples)),
            "error_ci_low": baseline_ci_low,
            "error_ci_high": baseline_ci_high,
            "feature_count": int(baseline_features.shape[0]),
        },
        "summary": enriched_rows,
    }
    with sidecar_path.open("w") as handle:
        json.dump(sidecar, handle, indent=2, default=roundtrip_jsonable)

    print(f"  wrote {summary_out_path}")
    print(f"  wrote {sidecar_path}")
    return summary_out_path, sidecar_path


def main() -> None:
    args = parse_args()
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError("--ci-level must be between 0 and 1")
    if args.fid_bootstrap_reps < 1:
        raise ValueError("--fid-bootstrap-reps must be at least 1")
    if args.fid_fold_permutations < 1:
        raise ValueError("--fid-fold-permutations must be at least 1")
    if args.fid_min_fold_size < 2:
        raise ValueError("--fid-min-fold-size must be at least 2")

    for result_dir in args.result_dirs:
        process_result_dir(
            result_dir=result_dir.resolve(),
            ci_level=args.ci_level,
            fid_interval_method=args.fid_interval_method,
            fid_bootstrap_reps=args.fid_bootstrap_reps,
            fid_fold_permutations=args.fid_fold_permutations,
            fid_min_fold_size=args.fid_min_fold_size,
            bootstrap_seed=args.bootstrap_seed,
            summary_suffix=args.summary_suffix,
            recompute_fid_point=args.recompute_fid_point,
        )


if __name__ == "__main__":
    main()
