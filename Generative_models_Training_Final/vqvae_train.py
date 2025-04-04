################# https://github.com/jiazhao97/VQ-VAE_withPixelCNNprior ############
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F 
import torch.optim as optim
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from dataset_loader.microalgae_dataset import MicroalgaeDataset
from utils import DatasetSampler
from models.vqvae.model import VQVAE

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_data_variance(data_iter):
    train_images = None
    for _ in tqdm(range(len(data_iter))):
        data = next(data_iter)
        images, _ = data
        if train_images is not None:
            train_images = np.vstack([train_images, images.numpy()])
        else:
            train_images = images.numpy()
            
    data_variance = np.var(train_images / 255.0)
    return data_variance

def evaluate(model, test_images, save_dir, iter):
    model.eval()
    x_recon = None
    with torch.no_grad():
        _, data_recon, _ = model(test_images)
        x_recon = data_recon
        
    model.train()
    # Add 0.5 to shift back the value to 0 - 1
    vutils.save_image(x_recon + 0.5, os.path.join(save_dir, f"iter_{iter}.png"), normalize=True)
    
def train(args):
    # Argparse Values
    dataset_path = args.dataset_path
    total_iterations = args.total_iterations
    batch_size = args.batch_size
    image_size = args.image_size
    save_dir = os.path.join(args.save_dir)
    save_interval = args.save_interval
    seed = args.seed
    checkpoint = args.ckpt
    dataloader_workers = args.workers
    
    # Model Config
    learning_rate = args.learning_rate
    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    beta = args.beta
    decay = args.decay
    epsilon = args.epsilon
    
    # Set seed
    torch.manual_seed(seed)

    current_iteration = 0
    saved_image_folder = os.path.join(save_dir, 'images')
    saved_model_folder = os.path.join(save_dir, 'models')
    
    # Create save directory if does not exist
    folders = [save_dir, saved_image_folder, saved_model_folder]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Transforms
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Normalized Pixel values to range of [0, 1]
        transforms.Normalize((0.5,0.5,0.5), (1.0, 1.0, 1.0)), #image = (image - mean) / std
    ]
    
    transforms_variance_list = [
        transforms.Resize((image_size, image_size)),
        transforms.PILToTensor()
    ]
    
    # Calculate Data Variance
    train_dataset_variance = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'train'), transforms=transforms_variance_list)
    train_dataloader_variance = iter(DataLoader(train_dataset_variance, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers))
    train_data_variance = calculate_data_variance(train_dataloader_variance)
    
    # Delete after calculation is done
    del train_dataloader_variance
    del train_dataset_variance
    
    print(f"Data Variance = {train_data_variance}")
    
    # Prepare Dataset
    train_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'train'), transforms=transforms_list)
    train_sampler = DatasetSampler(train_dataset)
    train_dataloader = iter(DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=dataloader_workers))
    
    # Prepare Test Dataset
    test_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path, 'test'), transforms=transforms_list)
    test_sampler = DatasetSampler(test_dataset)
    test_dataloader = iter(DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=dataloader_workers))
    
    # Prepare model
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              beta, decay, epsilon).to(DEVICE)
    
    # Optimizer
    # Original implementation does not have amsgrad. Keras tutorial set amsgrad=False as well.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False) 
    
    # Load checkpoint if exists
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        current_iteration = ckpt['iteration']
        del ckpt
    
    # Metrics
    loss_values = []
    
    progress_bar = tqdm(range(current_iteration, total_iterations + 1))
    with torch.autograd.set_detect_anomaly(True):
        for iteration in progress_bar:
            model.train()
            # Retrieve Training Data
            images, _ = next(train_dataloader)
            x  = images.to(DEVICE)
            
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(x)
            recon_error = F.mse_loss(data_recon, x) / train_data_variance
            loss = recon_error + vq_loss
            
            # Calculate Loss
            loss.backward()
            optimizer.step()
            
            # Store Metrics
            progress_bar.set_description('Loss: %.3f - Recon Error: %.3f - Perplexity: %.3f - VQ Loss: %.3f'%(loss, recon_error, perplexity, vq_loss))
            loss_values.append([iteration, loss.item(), recon_error.item(), perplexity.item(), vq_loss.item(), train_data_variance])
            
            if iteration % save_interval == 0 or iteration == total_iterations:
                # Sample Images
                test_images, _ = next(test_dataloader)
                evaluate(model, test_images.to(DEVICE), saved_image_folder, iteration)
                
            if iteration % save_interval == 0 or iteration == total_iterations:
                # Save Model Weights
                model_weights = model.state_dict()
                optimizer_state = optimizer.state_dict()
                torch.save({
                    'model': model_weights,
                    'optimizer': optimizer_state,
                    'iteration': iteration
                }, os.path.join(saved_model_folder, f'{iteration}.pth'))
                # Save loss
                df = pd.DataFrame(loss_values, columns=['Iteration', 'Loss', 'Recon Error', 'Perplexity', 'VQ Loss', 'Training Data Variance'])
                df.to_csv(os.path.join(save_dir, 'loss_per_iteration.csv'), index=False)
                
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQVAE')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', '1024x1024-0.8-v2'), help='Microalgae Training Dataset location')
    parser.add_argument('--total_iterations', type=int, default=5000, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--image_size', type=int, default=256, help='Image Resolution Size')
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'vqvae_train_results'), help='output results location')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of iterations to save model')
    parser.add_argument('--seed', type=int, default=42, help='PyTorch Seed')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
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
    
    train(args)