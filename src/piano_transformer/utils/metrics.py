import os
import glob
import numpy as np
import traceback
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
from miditok import REMI
from frechet_music_distance import FrechetMusicDistance
from symusic import Score
from piano_transformer.mgeval import core, utils
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

TOKENIZER_PATH = "/hpcwork/lect0148/experiments/mistral-162M_remi_maestro_v1/tokenizer.json" # adapt path for local use


def get_mgeval_features(num_samples):
    set_eval_init = {'total_used_pitch':np.zeros((num_samples,1)),
                     'total_pitch_class_histogram':np.zeros((num_samples,12)),
                     'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),
                     'pitch_range':np.zeros((num_samples,1)),
                     'avg_pitch_shift':np.zeros((num_samples,1)),
                     'total_used_note':np.zeros((num_samples,1)),
                     'avg_IOI':np.zeros((num_samples,1)),
                     'note_length_hist':np.zeros((num_samples, 12)),
                     'note_length_transition_matrix':np.zeros((num_samples, 12, 12)),
                    }
    kwargs_init = {"total_used_pitch": {},
                   "total_pitch_class_histogram": {},
                   "pitch_class_transition_matrix": {"normalize": 2},
                   "pitch_range": {},
                   "avg_pitch_shift": {"track_num": 0},
                   "total_used_note": {"track_num": 0},
                   "avg_IOI": {},
                   "note_length_hist": {"track_num": 0, "normalize": True, "pause_event": False},
                   "note_length_transition_matrix": {"track_num": 0, "normalize": 2, "pause_event": False}
                   }
    return set_eval_init, kwargs_init

def analyze_dataset_mgeval(dataset_path, features, max_samples=None):
    dataset = glob.glob(os.path.join(dataset_path, '*.midi'))
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
    num_samples = len(dataset)
    set_eval_init = {'total_used_pitch':np.zeros((num_samples,1)),
                 'total_used_note':np.zeros((num_samples,1)),
                 'total_pitch_class_histogram':np.zeros((num_samples,12)),
                 'pitch_range':np.zeros((num_samples,1)),
                 'avg_pitch_shift':np.zeros((num_samples,1))}
    set_eval = {key: set_eval_init[key] for key in features}
    metrics_list = features
    kwargs_init = {"total_used_pitch": {}, "total_used_note": {"track_num": 0}, "total_pitch_class_histogram": {}, "pitch_range": {}, "pitch_range": {"avg_pitch_shift": 0}}
    kwargs = [kwargs_init[key] for key in features]
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]}"):
            feature = core.extract_feature(dataset[i])
            set_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
            
    for i in range(0, len(metrics_list)):
        print('------------------------')
        print(metrics_list[i] + ':')
        print('mean: ', np.mean(set_eval[metrics_list[i]], axis=0))
        print('std: ', np.std(set_eval[metrics_list[i]], axis=0))
        

def comparing_pairwise_distances_mgeval(dataset1_path, dataset2_path, features, graphics_path, max_samples=None):
    dataset1 = glob.glob(os.path.join(dataset1_path, '*.midi'))
    dataset2 = glob.glob(os.path.join(dataset2_path, '*.midi'))
    if max_samples and len(dataset1) > max_samples:
        dataset1 = dataset1[:max_samples]
    if max_samples and len(dataset2) > max_samples:
        dataset2 = dataset2[:max_samples]
    num_samples = min(len(dataset1), len(dataset2))
    metrics_list = features
    
    set_eval_init = {'total_used_pitch':np.zeros((num_samples,1)),
                 'total_used_note':np.zeros((num_samples,1)),
                 'total_pitch_class_histogram':np.zeros((num_samples,12)),
                 'pitch_range':np.zeros((num_samples,1)),
                 'avg_pitch_shift':np.zeros((num_samples,1))}
    set1_eval = {key: set_eval_init[key] for key in features}
    set2_eval = copy.deepcopy(set1_eval)
    metrics_list = features
    kwargs_init = {"total_used_pitch": {}, "total_used_note": {"track_num": 0}, "total_pitch_class_histogram": {}, "pitch_range": {}, "pitch_range": {"avg_pitch_shift": 0}}
    kwargs = [kwargs_init[key] for key in features]
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset1"):
            feature = core.extract_feature(dataset1[i])
            set1_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset2"):
            feature = core.extract_feature(dataset2[i])
            set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
            
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    for i in range(len(metrics_list)):
        for train_index, test_index in tqdm(loo.split(np.arange(num_samples)), desc=f"Computing intra-set distances for {metrics_list[i]} on dataset1"):
            set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
    for i in range(len(metrics_list)):
        for train_index, test_index in tqdm(loo.split(np.arange(num_samples)), desc=f"Computing inter-set distances for {metrics_list[i]} between dataset1 and dataset2"):
            sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])
            
    plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)
    os.makedirs(graphics_path, exist_ok=True)
    for i in range(0,len(metrics_list)):
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.legend()
        output_path = os.path.join(graphics_path, f"{metrics_list[i]}_distance_plot.png")
        plt.savefig(output_path)
        plt.clf()
        
    for i in range(0, len(metrics_list)):
        print('------------------------')
        print( metrics_list[i] + ':')
        print('Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
        print('Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
    
    
        

    

def analyze_dataset_mgeval(dataset_path, features=None, max_samples=None):
    print("running full function")
    if not features:
        features = ['total_used_pitch', 'total_pitch_class_histogram', 'pitch_class_transition_matrix',
                    'pitch_range', 'avg_pitch_shift', 'total_used_note', 'avg_IOI', 'note_length_hist',
                    'note_length_transition_matrix']
    dataset = glob.glob(os.path.join(dataset_path, '*.midi'))
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
        print(dataset)
    num_samples = len(dataset)
    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
    set_eval = {key: set_eval_init[key] for key in features}
    kwargs = [kwargs_init[key] for key in features]
    metrics_list = features
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]}"):
            feature = core.extract_feature(dataset[i])
            set_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
            
    for i in range(0, len(metrics_list)):
        print('------------------------')
        print(metrics_list[i] + ':')
        print('mean: ', np.mean(set_eval[metrics_list[i]], axis=0))
        print('std: ', np.std(set_eval[metrics_list[i]], axis=0))

    # summarize_mgeval_results(set_eval, metrics_list)
    summarize_and_plot_mgeval_results(set_eval, metrics_list)

    
        

def comparing_pairwise_distances_mgeval(dataset1_path, dataset2_path, graphics_path, features=None, max_samples=None):
    print("running full function")
    if not features:
        # features = ['total_used_pitch', 'total_pitch_class_histogram', 'pitch_class_transition_matrix',
        #             'pitch_range', 'avg_pitch_shift', 'total_used_note', 'avg_IOI', 'note_length_hist',
        #             'note_length_transition_matrix']
        features = ['total_used_pitch', 'total_pitch_class_histogram',
                    'pitch_range', 'avg_pitch_shift', 'total_used_note', 'avg_IOI', 'note_length_hist',
                    'note_length_transition_matrix']
    dataset1 = glob.glob(os.path.join(dataset1_path, '*.midi'))
    dataset2 = glob.glob(os.path.join(dataset2_path, '*.midi'))
    if max_samples and len(dataset1) > max_samples:
        dataset1 = dataset1[:max_samples]
    if max_samples and len(dataset2) > max_samples:
        dataset2 = dataset2[:max_samples]
    num_samples = min(len(dataset1), len(dataset2))
    metrics_list = features
    
    set_eval_init, kwargs_init = get_mgeval_features(num_samples)
    set1_eval = {key: set_eval_init[key] for key in features}
    set2_eval = copy.deepcopy(set1_eval)
    kwargs = [kwargs_init[key] for key in features]
    metrics_list = features
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset1"):
            feature = core.extract_feature(dataset1[i])
            set1_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
    for j in range(len(metrics_list)):
        for i in tqdm(range(0, num_samples), desc=f"Evaluating {metrics_list[j]} on dataset2"):
            feature = core.extract_feature(dataset2[i])
            set2_eval[metrics_list[j]][i] = getattr(core.metrics(), metrics_list[j])(feature, **kwargs[j])
            
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    for i in range(len(metrics_list)):
        for train_index, test_index in tqdm(loo.split(np.arange(num_samples)), desc=f"Computing intra-set distances for {metrics_list[i]} on dataset1"):
            set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))
    for i in range(len(metrics_list)):
        for train_index, test_index in tqdm(loo.split(np.arange(num_samples)), desc=f"Computing inter-set distances for {metrics_list[i]} between dataset1 and dataset2"):
            sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])
            
    plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)
    os.makedirs(graphics_path, exist_ok=True)
    for i in range(0,len(metrics_list)):
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.legend()
        output_path = os.path.join(graphics_path, f"{metrics_list[i]}_distance_plot.png")
        plt.savefig(output_path)
        plt.clf()
        
    for i in range(0, len(metrics_list)):
        print('------------------------')
        print( metrics_list[i] + ':')
        print('Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
        print('Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
    
    
def summarize_mgeval_results(set_eval, metrics_list):
    summary = []

    for feature in metrics_list:
        mean_value = np.mean(set_eval[feature], axis=0)
        std_value = np.std(set_eval[feature], axis=0)

        if mean_value.ndim == 0 or mean_value.size == 1:
            mean_scalar = float(mean_value)
            std_scalar = float(std_value)
        else:
            mean_scalar = float(np.mean(mean_value))
            std_scalar = float(np.mean(std_value))

        summary.append({
            'Feature': feature,
            'Mean': mean_scalar,
            'Std': std_scalar
        })

    df = pd.DataFrame(summary)
    print(df.to_string(index=False))

    return df       


def summarize_and_plot_mgeval_results(set_eval, metrics_list):
    summary = []

    for feature in metrics_list:
        mean_value = np.mean(set_eval[feature], axis=0)
        std_value = np.std(set_eval[feature], axis=0)

        # Case 1 — Scalar features:
        if mean_value.ndim == 0 or mean_value.size == 1:
            mean_scalar = float(mean_value)
            std_scalar = float(std_value)
            summary.append({'Feature': feature, 'Mean': mean_scalar, 'Std': std_scalar})

        # Case 2 — Vector features: pitch_class_histogram and note_length_hist
        elif mean_value.ndim == 1:

            if feature == 'total_pitch_class_histogram':
                labels = ['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B']
                plt.figure(figsize=(8, 4))
                plt.bar(labels, mean_value)
                plt.title(f'Pitch Class Histogram ({feature})')
                plt.ylabel('Proportion')
                plt.show()

            elif feature == 'note_length_hist':
                labels = [
                    "Full", "Half", "Quarter", "8th", "16th",
                    "Dot Half", "Dot Quarter", "Dot 8th", "Dot 16th",
                    "Half Triplet", "Quarter Triplet", "8th Triplet"
                ]
                plt.figure(figsize=(10, 4))
                plt.bar(labels, mean_value)
                plt.title(f'Note Length Histogram ({feature})')
                plt.ylabel('Proportion')
                plt.xticks(rotation=45)
                plt.show()

            # Still report simple global mean/std for table:
            mean_scalar = float(np.mean(mean_value))
            std_scalar = float(np.mean(std_value))
            summary.append({'Feature': feature, 'Mean': mean_scalar, 'Std': std_scalar})

        # Case 3 — Matrix features: transition matrices
        elif mean_value.ndim == 2:
            plt.figure(figsize=(8, 6))
            sns.heatmap(mean_value, annot=False, cmap='viridis')
            plt.title(f'Heatmap ({feature})')
            plt.show()

            # Again, report mean of all matrix values
            mean_scalar = float(np.mean(mean_value))
            std_scalar = float(np.mean(std_value))
            summary.append({'Feature': feature, 'Mean': mean_scalar, 'Std': std_scalar})

        else:
            raise ValueError(f"Unhandled feature shape: {feature}")

    # Print global summary table:
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))

    return df
    

def extract_features(
    directory,
    tokenizer_path=TOKENIZER_PATH,
    features=("pitch", "duration", "velocity")
):
    tokenizer = REMI(params=tokenizer_path)
    pitch_vals, duration_vals, velocity_vals = [], [], []

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".midi"):
                continue

            midi_path = os.path.join(root, file)
            try:
                midi = Score(midi_path)
                token_ids = tokenizer(midi)

                # Take only the piano track
                if isinstance(token_ids, list):
                    token_ids = token_ids[0]

                # Convert token IDs to string tokens
                vocab_inv = {v: k for k, v in tokenizer.vocab.items()}
                tokens = [vocab_inv[i] for i in token_ids if i in vocab_inv]

            except Exception as e:
                print(f"Error tokenizing {file}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            for tok in tokens:
                if "pitch" in features and "Pitch_" in tok:
                    try:
                        pitch_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue
                if "duration" in features and "Duration_" in tok:
                    try:
                        duration_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue
                if "velocity" in features and "Velocity_" in tok:
                    try:
                        velocity_vals.append(int(tok.split("_")[-1]))
                    except:
                        continue

    return {
        "pitch": pitch_vals,
        "duration": duration_vals,
        "velocity": velocity_vals
    }

def compute_pdf(values, num_points=1000):
    values = np.array(values).astype(np.float64)
    if len(np.unique(values)) < 2:
        return None  # Cannot build KDE from constant data

    kde = gaussian_kde(values)
    x = np.linspace(np.min(values), np.max(values), num_points)
    pdf = kde.evaluate(x)
    pdf /= np.sum(pdf)  # Normalize
    return pdf


# Kullback-Leibler Divergence
def kld(p, q, epsilon=1e-10):
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(rel_entr(p, q))


# Overlapping Area
def oa(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(np.minimum(p, q))


# Frechet Music Distance
def fmd(ref_dir, gen_dir):
    fmd_metric = FrechetMusicDistance(
        feature_extractor='clamp2',
        gaussian_estimator='mle', 
        verbose=True
    )

    return fmd_metric.score(
        reference_path=ref_dir,
        test_path=gen_dir
    )


def compare(ref_dir, gen_dir, metric, tokenizer_path=TOKENIZER_PATH, feature="pitch"):
    if metric == fmd:
        print("Using FMD (input feature ignored)")
        return fmd(ref_dir, gen_dir)

    ref_feats = extract_features(ref_dir, tokenizer_path, features=(feature,))
    gen_feats = extract_features(gen_dir, tokenizer_path, features=(feature,))

    ref_vals = ref_feats.get(feature, [])
    gen_vals = gen_feats.get(feature, [])

    if len(ref_vals) < 2 or len(gen_vals) < 2:
        return None

    ref_pdf = compute_pdf(ref_vals)
    gen_pdf = compute_pdf(gen_vals)

    if ref_pdf is None or gen_pdf is None or len(ref_pdf) != len(gen_pdf):
        return None

    return metric(ref_pdf, gen_pdf)
