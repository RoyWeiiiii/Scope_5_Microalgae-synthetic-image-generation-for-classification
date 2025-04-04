################## t-SNE 2D plot visualisation with Resnet-50 (with Downsampling features) #########################
import os
import argparse
import torch
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data.dataloader import DataLoader
from dataset_loader.microalgae_dataset import MicroalgaeDataset
import time
from collections import defaultdict

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base path to all datasets
BASE_DATASET_PATH = r"D:\\CodingProjects\\machine_learning\\Experiment_5_Latest\\datasets"

SELECTED_DATASET_PATHS = [
    os.path.join(BASE_DATASET_PATH, 'original_datasets', '1024x1024-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'cond_ddim-256x256-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'ddim-256x256-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'fastgan-256x256-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'cond_fastgan-256x256-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'pixelcnn_cond_vqvae-256x256-0.8-v2'),
    os.path.join(BASE_DATASET_PATH, 'generated_datasets', 'pixelcnn_vqvae-256x256-0.8-v2'),
]

DATASET_NAME_REPLACEMENTS = {
    '1024x1024-0.8-v2': 'Original',
    'cond_ddim-256x256-0.8-v2': 'Conditional DDIM',
    'ddim-256x256-0.8-v2': 'DDIM',
    'cond_fastgan-256x256-0.8-v2': 'Conditional FastGAN',
    'fastgan-256x256-0.8-v2': 'FastGAN',
    'pixelcnn_cond_vqvae-256x256-0.8-v2': 'PixelCNN + Conditional VQVAE',
    'pixelcnn_vqvae-256x256-0.8-v2': 'PixelCNN + VQVAE'
}

# Updated colorblind-friendly palette
COLORS_PER_CLASS = {
    'Original - Chlamydomonas': (31, 119, 180),
    'Original - FSP-E': (174, 199, 232),
    'Original - Spirulina': (255, 127, 14),
    'Conditional FastGAN - Chlamydomonas': (255, 187, 120),
    'Conditional FastGAN - FSP-E': (44, 160, 44),
    'Conditional FastGAN - Spirulina': (152, 223, 138),
    'FastGAN - Chlamydomonas': (214, 39, 40),
    'FastGAN - FSP-E': (255, 152, 150),
    'FastGAN - Spirulina': (148, 103, 189),
    'Conditional DDIM - Chlamydomonas': (197, 176, 213),
    'Conditional DDIM - FSP-E': (140, 86, 75),
    'Conditional DDIM - Spirulina': (196, 156, 148),
    'DDIM - Chlamydomonas': (227, 119, 194),
    'DDIM - FSP-E': (247, 182, 210),
    'DDIM - Spirulina': (127, 127, 127),
    'PixelCNN + Conditional VQVAE - Chlamydomonas': (199, 199, 199),
    'PixelCNN + Conditional VQVAE - FSP-E': (188, 189, 34),
    'PixelCNN + Conditional VQVAE - Spirulina': (219, 219, 141),
    'PixelCNN + VQVAE - Chlamydomonas': (23, 190, 207),
    'PixelCNN + VQVAE - FSP-E': (158, 218, 229),
    'PixelCNN + VQVAE - Spirulina': (31, 119, 180),
}

# Function to selectively subsample specific model-class combinations (Model-class downsampling)
def subsample_specific_model_classes(features, feature_labels, sample_ratio_map):
    """
    Downsamples only specified model-class combinations with individual ratios.
    sample_ratio_map: dict like {
        'FastGAN - Spirulina': 0.3,
        'Conditional FastGAN - Chlamydomonas': 0.1
    }
    """
    label_to_indices = defaultdict(list)
    for i, label in enumerate(feature_labels):
        label_to_indices[label].append(i)

    selected_indices = []
    for label, indices in label_to_indices.items():
        if label in sample_ratio_map:
            ratio = sample_ratio_map[label]
            n_select = max(1, int(len(indices) * ratio))
            sampled = np.random.choice(indices, n_select, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)

    selected_indices = sorted(selected_indices)
    features = features[selected_indices]
    feature_labels = [feature_labels[i] for i in selected_indices]
    return features, feature_labels


MARKERS_PER_CLASS = {
    'Original - Spirulina': 'o',
    'Original - Chlamydomonas': 'o',
    'Original - FSP-E': 'o',
    'Conditional FastGAN - Spirulina': 'v',
    'Conditional FastGAN - Chlamydomonas': 'v',
    'Conditional FastGAN - FSP-E': 'v',
    'FastGAN - Spirulina': '>',
    'FastGAN - Chlamydomonas': '>',
    'FastGAN - FSP-E': '>',
    'Conditional DDIM - Spirulina': 's',
    'Conditional DDIM - Chlamydomonas': 's',
    'Conditional DDIM - FSP-E': 's',
    'DDIM - Spirulina': '<',
    'DDIM - Chlamydomonas': '<',
    'DDIM - FSP-E': '<',
    'PixelCNN + Conditional VQVAE - Spirulina': '^',
    'PixelCNN + Conditional VQVAE - Chlamydomonas': '^',
    'PixelCNN + Conditional VQVAE - FSP-E': '^',
    'PixelCNN + VQVAE - Chlamydomonas': 'D',
    'PixelCNN + VQVAE - FSP-E': 'D',
    'PixelCNN + VQVAE - Spirulina': 'D',
}

LABEL_TO_CLASS = {
    0: 'Chlamydomonas',
    1: 'FSP-E',
    2: 'Spirulina'
}

COMBINED_GRAPH_CONFIG = {
    'Original Datasets': {
        'datasets': [
            'Original - Chlamydomonas',
            'Original - Spirulina',
            'Original - FSP-E'
        ]
    },
    'FastGAN Variants': {
        'datasets': [
            'Original - Chlamydomonas',
            'Original - Spirulina',
            'Original - FSP-E',
            'Conditional FastGAN - Chlamydomonas',
            'Conditional FastGAN - FSP-E',
            'Conditional FastGAN - Spirulina',
            'FastGAN - Chlamydomonas',
            'FastGAN - FSP-E',
            'FastGAN - Spirulina',
        ]
    },
    'DDIM Variants': {
        'datasets': [
            'Original - Chlamydomonas',
            'Original - Spirulina',
            'Original - FSP-E',
            'Conditional DDIM - Chlamydomonas',
            'Conditional DDIM - FSP-E',
            'Conditional DDIM - Spirulina',
            'DDIM - Chlamydomonas',
            'DDIM - FSP-E',
            'DDIM - Spirulina',
         ]
     },
    'PixelCNN+VQVAE Variants': {
        'datasets': [
            'Original - Chlamydomonas',
            'Original - Spirulina',
            'Original - FSP-E',
            'PixelCNN + Conditional VQVAE - Chlamydomonas',
            'PixelCNN + Conditional VQVAE - FSP-E',
            'PixelCNN + Conditional VQVAE - Spirulina',
            'PixelCNN + VQVAE - Chlamydomonas',
            'PixelCNN + VQVAE - FSP-E',
            'PixelCNN + VQVAE - Spirulina',
        ]
    },
    'Combined Datasets': {
        'datasets': [
            'Original - Chlamydomonas',
            'Original - FSP-E',
            'Original - Spirulina',
            'Conditional FastGAN - Chlamydomonas',
            'Conditional FastGAN - FSP-E',
            'Conditional FastGAN - Spirulina',
            'FastGAN - Chlamydomonas',
            'FastGAN - FSP-E',
            'FastGAN - Spirulina',
            'Conditional DDIM - Chlamydomonas',
            'Conditional DDIM - FSP-E',
            'Conditional DDIM - Spirulina',
            'DDIM - Chlamydomonas',
            'DDIM - FSP-E',
            'DDIM - Spirulina',
            'PixelCNN + Conditional VQVAE - Chlamydomonas',
            'PixelCNN + Conditional VQVAE - FSP-E',
            'PixelCNN + Conditional VQVAE - Spirulina',
            'PixelCNN + VQVAE - Chlamydomonas',
            'PixelCNN + VQVAE - FSP-E',
            'PixelCNN + VQVAE - Spirulina'
        ]
    },
}

def scale_to_01_range(x):
    value_range = np.max(x) - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def generate_chart(tx, ty, feature_labels, save_dir):
    progress_bar = tqdm(enumerate(COMBINED_GRAPH_CONFIG))
    for index, graph_type in progress_bar:
        datasets = COMBINED_GRAPH_CONFIG[graph_type]['datasets']
        fig, ax = plt.subplots(figsize=(10.7, 12.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both')
        plt.axis('equal')

        for label in datasets:
            progress_bar.set_description(f"{graph_type} ==== {label}")
            indices = [i for i, current_label in enumerate(feature_labels) if current_label == label]
            if not indices:
                continue
            current_tx, current_ty = np.take(tx, indices), np.take(ty, indices)
            ax.scatter(
                current_tx, current_ty,
                c=[tuple(color / 255 for color in COLORS_PER_CLASS[label])] * current_tx.shape[0],
                marker=MARKERS_PER_CLASS[label],
                label=label,
                edgecolor='black',
                linewidth=1
            )

        plt.title(f'{graph_type}')
        plt.subplots_adjust(bottom=0.5 if graph_type == 'Combined Datasets' else 0.3)
        # fig.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), borderpad=1.6)  #Toggle the legend#
        output_path = os.path.join(save_dir, f"TSNE - {graph_type}.png")
        fig.savefig(output_path)
        print(f"✅ Saved: {output_path}")
        plt.close()

def generate_subcharts(tx, ty, feature_labels, save_dir):
    plt.rcParams.update({'font.size': 20})
    progress_bar = tqdm(enumerate(COMBINED_GRAPH_CONFIG))

    for index, graph_type in progress_bar:
        config = COMBINED_GRAPH_CONFIG[graph_type]
        datasets = config['datasets']
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_title(f'{graph_type}', fontsize=24)
        plt.axis('equal')

        for label in datasets:
            progress_bar.set_description(f"{graph_type} ==== {label}")
            indices = [i for i, current_label in enumerate(feature_labels) if current_label == label]
            if not indices:
                continue
            current_tx, current_ty = np.take(tx, indices), np.take(ty, indices)
            ax.scatter(
                current_tx, current_ty,
                c=[tuple(color / 255 for color in COLORS_PER_CLASS[label])] * len(current_tx),
                marker=MARKERS_PER_CLASS[label],
                s=120,
                label=label,
                edgecolor='black',
                linewidth=1
            )

        # ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=14) # Toggle the legends #
        fig.tight_layout()
        # fig.tight_layout(rect=[0, 0.05, 1, 1])                                          # Toggle the legends #
        save_file = os.path.join(save_dir, f"Subplot - {graph_type}.png")
        fig.savefig(save_file)
        print(f"✅ Saved: {save_file}")
        plt.close()

def generate(args):
    batch_size = args.batch_size
    save_dir = os.path.abspath(args.save_dir)
    seed = args.seed
    dataloader_workers = args.workers

    print(f"📦 Batch size: {batch_size}, 🧵 Workers: {dataloader_workers}, 🌱 Seed: {seed}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 Created save directory: {save_dir}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(DEVICE)
    model.eval()

    features = None
    feature_labels = []

    with torch.no_grad():
        for dataset_path in SELECTED_DATASET_PATHS:
            print(f"📂 Processing dataset: {dataset_path}")
            train_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'train'), weight_transforms=weights.transforms())
            test_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'test'), weight_transforms=weights.transforms())
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
            dataloader = iter(DataLoader(dataset, batch_size=batch_size, num_workers=dataloader_workers))
            progress_bar = tqdm(dataloader, desc="Extracting Batches")

            for images, labels in progress_bar:
                images = images.to(DEVICE)
                output = model(images)
                output = torch.flatten(output, 1)
                output = output.detach().cpu().numpy()
                label_list = labels.detach().cpu().tolist()
                dataset_label = DATASET_NAME_REPLACEMENTS[os.path.basename(dataset_path)]
                label_names = [f"{dataset_label} - {LABEL_TO_CLASS[label]}" for label in label_list]
                feature_labels += label_names
                features = output if features is None else np.concatenate((features, output))

    print("🔍 Found labels:", set(feature_labels))
    print("🧮 Total samples:", len(feature_labels))
    
    # Downsample based on models-class chooose [sample_ratio=1.0 Keep 100% features] -Adjust this
    sample_ratio_map = {
    'FastGAN - FSP-E': 0.3,
    'FastGAN - Spirulina': 0.3,
    'Conditional FastGAN - FSP-E': 0.3,
    'DDIM - Spirulina': 0.3,
    }
    features, feature_labels = subsample_specific_model_classes(features, feature_labels, sample_ratio_map)

    print("🎯 Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=seed).fit_transform(features)
    tx, ty = scale_to_01_range(tsne[:, 0]), scale_to_01_range(tsne[:, 1])

    print("📊 Generating individual t-SNE charts by model type...")
    generate_chart(tx, ty, feature_labels, save_dir)

    print("📌 Generating separate subplots (one per graph type)...")
    generate_subcharts(tx, ty, feature_labels, save_dir)

    elapsed = time.time() - start_time
    print(f"🏁 All visualizations saved. Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--save_dir', type=str, default=os.path.join(BASE_DATASET_PATH, 'tsne_plots_original_VQVAE_color_downsample_V1'), help='Directory to save all chart images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')

    args = parser.parse_args()
    print(args)
    generate(args)