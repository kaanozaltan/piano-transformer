"""
Simplified metrics module for evaluation without FMD dependencies
Contains create_subset and evaluate_mgeval_combined functions
"""

import os
import glob
import numpy as np
import traceback
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pretty_midi
import pickle
import shutil
import hashlib
import random
from tqdm import tqdm

# Try to import mgeval - gracefully handle if not available
try:
    import sys
    # Add src to path to find piano_transformer.mgeval
    current_dir = os.path.dirname(__file__)
    src_path = os.path.abspath(os.path.join(current_dir, '../../../..', 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from piano_transformer.mgeval import core, utils
    MGEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: mgeval not available: {e}")
    MGEVAL_AVAILABLE = False


def get_mgeval_features(num_samples):
    """Initialize mgeval feature containers"""
    set_eval_init = {
        "total_used_pitch": np.zeros((num_samples, 1)),
        "total_pitch_class_histogram": np.zeros((num_samples, 12)),
        "pitch_class_transition_matrix": np.zeros((num_samples, 12, 12)),
        "pitch_range": np.zeros((num_samples, 1)),
        "avg_pitch_shift": np.zeros((num_samples, 1)),
        "total_used_note": np.zeros((num_samples, 1)),
        "avg_IOI": np.zeros((num_samples, 1)),
        "note_length_hist": np.zeros((num_samples, 32)),
        "note_length_transition_matrix": np.zeros((num_samples, 32, 32)),
    }

    kwargs_init = {
        "total_used_pitch": {},
        "total_pitch_class_histogram": {"normalize": True},
        "pitch_class_transition_matrix": {"normalize": True},
        "pitch_range": {},
        "avg_pitch_shift": {},
        "total_used_note": {},
        "avg_IOI": {},
        "note_length_hist": {"normalize": True},
        "note_length_transition_matrix": {"normalize": True},
    }

    return set_eval_init, kwargs_init


def summarize_and_plot_mgeval_results(set_eval, metrics_list, dataset_name, output_path=None):
    """Summarize mgeval results without plotting to avoid matplotlib issues"""
    summary = []
    
    for metric in metrics_list:
        if metric not in set_eval:
            continue
            
        data = set_eval[metric]
        
        # Handle different data shapes
        if data.ndim == 1:
            values = data[~np.isnan(data)]
        elif data.ndim == 2 and data.shape[1] == 1:
            values = data[:, 0]
            values = values[~np.isnan(values)]
        else:
            # For matrices, flatten and use
            values = data.flatten()
            values = values[~np.isnan(values)]
        
        if len(values) == 0:
            continue
            
        summary.append({
            "Feature": metric,
            "Mean": np.mean(values),
            "Std": np.std(values),
            "Min": np.min(values),
            "Max": np.max(values),
        })
    
    return summary


def compare_mgeval_distributions(set1_eval, set2_eval, metrics_list, output_path=None):
    """Compare distributions between two datasets"""
    summary = []
    
    for metric in metrics_list:
        if metric not in set1_eval or metric not in set2_eval:
            continue
            
        data1 = set1_eval[metric]
        data2 = set2_eval[metric]
        
        # Handle different data shapes
        if data1.ndim == 1:
            values1 = data1[~np.isnan(data1)]
            values2 = data2[~np.isnan(data2)]
        elif data1.ndim == 2 and data1.shape[1] == 1:
            values1 = data1[:, 0]
            values1 = values1[~np.isnan(values1)]
            values2 = data2[:, 0]
            values2 = values2[~np.isnan(values2)]
        else:
            # For matrices, flatten and use
            values1 = data1.flatten()
            values1 = values1[~np.isnan(values1)]
            values2 = data2.flatten()
            values2 = values2[~np.isnan(values2)]
        
        if len(values1) == 0 or len(values2) == 0:
            continue
        
        # Compute KLD and OA
        try:
            # Simple histogram-based approach
            bins = np.linspace(min(np.min(values1), np.min(values2)), 
                             max(np.max(values1), np.max(values2)), 50)
            hist1, _ = np.histogram(values1, bins=bins, density=True)
            hist2, _ = np.histogram(values2, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # KLD
            kld = np.sum(rel_entr(hist1, hist2))
            
            # OA (Overlap Area)
            oa = np.sum(np.minimum(hist1, hist2))
            
            summary.append({
                "Feature": metric,
                "KLD": kld if not np.isnan(kld) else 0.0,
                "OA": oa if not np.isnan(oa) else 0.0,
            })
            
        except Exception as e:
            print(f"Error computing metrics for {metric}: {e}")
            continue
    
    return summary


def evaluate_mgeval_combined(dataset1_path, dataset2_path, output_path=None, features=None, max_samples=None):
    """
    Evaluate two datasets using mgeval metrics
    Returns absolute and relative summaries
    """
    if not MGEVAL_AVAILABLE:
        print("Warning: mgeval not available, returning empty results")
        return [], []
    
    print("Running combined MGEval evaluation (absolute + relative)...")

    if not features:
        features = [
            "total_used_pitch",
            "total_pitch_class_histogram",
            "pitch_range",
            "avg_pitch_shift",
            "total_used_note",
            "avg_IOI",
            "note_length_hist",
            "note_length_transition_matrix",
        ]

    # Load and filter datasets
    def load_valid_dataset(dataset_path):
        dataset = glob.glob(os.path.join(dataset_path, "**", "*.mid*"), recursive=True)
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]
        valid_dataset = []
        for path in dataset:
            try:
                midi = pretty_midi.PrettyMIDI(path)
                if midi.instruments:
                    valid_dataset.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return valid_dataset

    dataset1 = load_valid_dataset(dataset1_path)
    dataset2 = load_valid_dataset(dataset2_path)

    num_samples = min(len(dataset1), len(dataset2))
    dataset1 = dataset1[:num_samples]
    dataset2 = dataset2[:num_samples]

    if num_samples == 0:
        print("No valid MIDI files found in at least one dataset.")
        return [], []

    print(f"Using {num_samples} samples from each dataset")

    # Extract features
    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
    set1_eval = {key: set_eval_init[key].copy() for key in features}
    set2_eval = {key: set_eval_init[key].copy() for key in features}
    kwargs = {key: kwargs_init[key] for key in features}

    for j, feature_name in enumerate(features):
        print(f"Extracting {feature_name}")
        
        # Dataset 1
        for i in tqdm(range(num_samples), desc=f"  Dataset 1"):
            try:
                feature = core.extract_feature(dataset1[i])
                value = getattr(core.metrics(), feature_name)(feature, **kwargs[feature_name])

                # Check for NaNs in output and skip if found
                if np.any(np.isnan(value)):
                    raise ValueError("NaN in extracted feature")

                set1_eval[feature_name][i] = value

            except Exception as e:
                print(f"[{feature_name}] Skipping {dataset1[i]} (idx={i}): {e}")
                continue
                
        # Dataset 2
        for i in tqdm(range(num_samples), desc=f"  Dataset 2"):
            try:
                feature = core.extract_feature(dataset2[i])
                value = getattr(core.metrics(), feature_name)(feature, **kwargs[feature_name])

                # Check for NaNs in output and skip if found
                if np.any(np.isnan(value)):
                    raise ValueError("NaN in extracted feature")

                set2_eval[feature_name][i] = value

            except Exception as e:
                print(f"[{feature_name}] Skipping {dataset2[i]} (idx={i}): {e}")
                continue

    # Generate summaries
    print("\nAbsolute Evaluation: Dataset 1")
    absolute_summary_train = summarize_and_plot_mgeval_results(
        set1_eval, features, "dataset1", output_path
    )
    print("\nAbsolute Evaluation: Dataset 2")
    absolute_summary_generated = summarize_and_plot_mgeval_results(
        set2_eval, features, "dataset2", output_path
    )

    # Compute relative differences
    absolute_summary = []
    for i, feature in enumerate(features):
        if i < len(absolute_summary_train) and i < len(absolute_summary_generated):
            train_mean = absolute_summary_train[i]["Mean"]
            gen_mean = absolute_summary_generated[i]["Mean"]
            
            rel_diff_mean = (gen_mean - train_mean) / (train_mean + 1e-10) if train_mean != 0 else 0
            rel_diff_std = 0  # Simplified
            
            absolute_summary.append({
                "Feature": feature,
                "Rel_Diff_Mean": rel_diff_mean,
                "Rel_Diff_Std": rel_diff_std,
            })

    print("\nRelative Evaluation")
    relative_summary = compare_mgeval_distributions(
        set1_eval, set2_eval, features, output_path
    )

    return absolute_summary, relative_summary


def create_subset(input_dir, subset_size, seed=None):
    """Create a deterministic subset of MIDI files from a directory"""
    parent_dir = os.path.dirname(os.path.abspath(input_dir))
    base_name = os.path.basename(os.path.normpath(input_dir))

    # Determine effective seed
    if seed is None:
        # Use hash-based deterministic seed
        all_files = sorted(
            glob.glob(os.path.join(input_dir, "**", "*.mid*"), recursive=True),
            key=os.path.basename,
        )
        relative_names = sorted([os.path.basename(path) for path in all_files])
        seed_input = "".join(relative_names)
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (10**8)
        print(f"Generated deterministic seed: {seed}")
    elif seed == -1:
        # Use true randomness
        seed = random.randint(0, 10**8)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")

    output_dir = os.path.join(parent_dir, f"{base_name}_subset_{subset_size}")

    if os.path.exists(output_dir):
        print("Subset already exists.")
        return output_dir

    all_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.mid*"), recursive=True),
        key=os.path.basename,
    )

    if len(all_files) < subset_size:
        print("Subset size exceeds available files.")
        return input_dir

    rng = random.Random(seed)
    subset = rng.sample(all_files, subset_size)

    os.makedirs(output_dir)
    for src in subset:
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Created subset at {output_dir}.")
    return output_dir