import os
import argparse
import torch
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset_loader.microalgae_dataset import MicroalgaeDataset
from constants.labels import CLASS_NAME_TO_LABELS_MAP
from utils import DatasetSampler
from models.class_conditioned_vqvae.model import VQVAE

def evaluate(args):
    dataset_path = args.dataset_path
    model_weight = args.model_weight
    save_path = args.save_path
    batch_size = args.batch_size
    image_size = args.image_size
    num_iterations = args.num_iterations
    num_classes = len(CLASS_NAME_TO_LABELS_MAP)
    dataloader_workers = args.workers
    
    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    beta = args.beta
    decay = args.decay
    epsilon = args.epsilon
    
    # Specify Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WEIGHTS_PATH = model_weight
    SAVE_DIR = save_path
    LABEL_TO_CLASS = {v: k for k, v in CLASS_NAME_TO_LABELS_MAP.items()}
    
    for key in CLASS_NAME_TO_LABELS_MAP.keys():
        if not os.path.exists(os.path.join(SAVE_DIR, key)):
            os.makedirs(os.path.join(SAVE_DIR, key))
            
    # Load Model Weights
    checkpoint = torch.load(WEIGHTS_PATH)
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              beta, decay, epsilon, num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Transforms
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Normalized Pixel values to range of [0, 1]
        transforms.Normalize((0.5,0.5,0.5), (1.0, 1.0, 1.0)), #image = (image - mean) / std
    ]
    
    # Prepare Test Dataset
    test_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'test'), transforms=transforms_list)
    test_sampler = DatasetSampler(test_dataset)
    test_dataloader = iter(DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=dataloader_workers))
    
    progress_bar = tqdm(range(num_iterations + 1))
    
    with torch.no_grad():
        for iteration in progress_bar:
            images, labels = next(test_dataloader)
            x, y = images.to(DEVICE), labels.to(DEVICE)
            _, data_recon, _ = model(x, y)
            
            for index, x_recon in enumerate(data_recon):
                curr_label = y[index].cpu().numpy().item()
                class_name = LABEL_TO_CLASS[curr_label]
                save_image(x_recon + 0.5, os.path.join(SAVE_DIR, class_name, f"{(iteration * batch_size) + index}.png"), normalize=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', 'original_datasets', '1024x1024-0.8-v2'), help='Microalgae Training Dataset location')
    parser.add_argument('--model_weight', type=str, default=os.path.join('model_weights', 'conditional_vqvae_results', 'models', '5000.pth'), help='Model weight path')
    parser.add_argument('--save_path', type=str, default=os.path.join('temp', 'conditional_vqvae_eval_results'), help='DEFAULT: Save generated image folder')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    
    parser.add_argument('--num_hiddens', type=int, default=128, help='Hidden Layer Size')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Residual Hidden Size')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='Number of Residual Layers')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding Dimension')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Embedding Dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment Cost or Beta')
    parser.add_argument('--decay', type=float, default=0, help='Exponential Moving Average decay value')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Exponential Moving Average epsilon value')
    
    args = parser.parse_args()
    print(args)
    evaluate(args)