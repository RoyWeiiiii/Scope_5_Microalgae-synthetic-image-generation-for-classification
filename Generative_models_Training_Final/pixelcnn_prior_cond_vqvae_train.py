#####################  https://github.com/jiazhao97/VQ-VAE_withPixelCNNprior #################

import argparse
import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from dataset_loader.microalgae_dataset import MicroalgaeDataset
from constants.labels import CLASS_NAME_TO_LABELS_MAP
from utils import DatasetSampler
from models.class_conditioned_vqvae.model import VQVAE
from models.gated_pixelcnn.model import GatedPixelCNN

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, pixelcnn_prior, pixelcnn_input_shape, image_shape, batch_size, save_dir, iter):
    with torch.no_grad():
        labels = torch.Tensor([0, 1, 2]).to(DEVICE).long()
        
        priors = pixelcnn_prior.generate(labels, shape=image_shape, batch_size=batch_size)
        prior_images = priors.cpu().data.float() / (pixelcnn_input_shape - 1)
        
        vutils.save_image(prior_images[:, None], os.path.join(save_dir, f"pixelcnn_iter_{iter}.png"), normalize=True)
        
        images = model.decode(priors, labels)
        vutils.save_image(images, os.path.join(save_dir, f"pixelcnn_decoded_iter_{iter}.png"), normalize=True)


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
    learning_rate = args.learning_rate
    
    # VQVAE Config
    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    beta = args.beta
    decay = args.decay
    epsilon = args.epsilon
    vqvae_checkpoint = args.vqvae_ckpt
    
    # PixelCNN
    pixelcnn_hidden_size_prior = args.pixelcnn_hidden_size_prior
    pixelcnn_num_layers = args.pixelcnn_num_layers
    pixelcnn_checkpoint = args.ckpt
    
    # Set seed
    torch.manual_seed(seed)
    
    num_classes = len(CLASS_NAME_TO_LABELS_MAP)
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
              beta, decay, epsilon, num_classes).to(DEVICE)
    
    # Load VQVAE weights
    vqvae_ckpt = torch.load(vqvae_checkpoint)
    model.load_state_dict(vqvae_ckpt['model'])
    model.eval()
    
    # GatedPixelCNN
    pixelcnn_prior = GatedPixelCNN(num_embeddings, pixelcnn_hidden_size_prior, pixelcnn_num_layers, num_classes).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(pixelcnn_prior.parameters(), lr=learning_rate)
    
    # Load PixelCNN Checkpoint if Exists
    if checkpoint != 'None':
        ckpt = torch.load(pixelcnn_checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        current_iteration = ckpt['iteration']
        del ckpt
    
    # Metrics
    loss_values = []
    
    progress_bar = tqdm(range(current_iteration, total_iterations + 1))
    for iteration in progress_bar:
        images, labels = next(train_dataloader)
        x, y = images.to(DEVICE), labels.to(DEVICE)
        
        with torch.no_grad():
            latents = model.encode(x, y)
            latents = latents.detach()
        
        logits = pixelcnn_prior(latents, y)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        
        # Calculate Loss and backprop
        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, num_embeddings), latents.view(-1))
        loss.backward()
        optimizer.step()
        
        # Store Metrics
        progress_bar.set_description('Loss: %.3f'%(loss))
        loss_values.append([iteration, loss.item()])
        
        if iteration % save_interval == 0 or iteration == total_iterations:
            # Sample Images
            evaluate(model, pixelcnn_prior, num_embeddings, (64, 64), 3, saved_image_folder, iteration)
                
        if iteration % save_interval == 0 or iteration == total_iterations:
            # Save Model Weights
            model_weights = pixelcnn_prior.state_dict()
            optimizer_state = optimizer.state_dict()
            torch.save({
                'model': model_weights,
                'optimizer': optimizer_state,
                'iteration': iteration
            }, os.path.join(saved_model_folder, f'{iteration}.pth'))
            # Save loss
            df = pd.DataFrame(loss_values, columns=['Iteration', 'Loss'])
            df.to_csv(os.path.join(save_dir, 'loss_per_iteration.csv'), index=False)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQVAE')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', '1024x1024-0.8-v2'), help='Microalgae Training Dataset location')
    parser.add_argument('--total_iterations', type=int, default=5000, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--image_size', type=int, default=256, help='Image Resolution Size')
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'pixelcnn_conditional_vqvae_train_results'), help='output results location')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of iterations to save model')
    parser.add_argument('--seed', type=int, default=42, help='PyTorch Seed')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    
    #VQVAE
    parser.add_argument('--num_hiddens', type=int, default=128, help='VQVAE Hidden Layer Size')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='VQVAE Residual Hidden Size')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='VQVAE Number of Residual Layers')
    parser.add_argument('--num_embeddings', type=int, default=512, help='VQVAE Embedding Dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='VQVAE Commitment Cost or Beta')
    parser.add_argument('--decay', type=float, default=0.99, help='VQVAE Exponential Moving Average decay value')
    parser.add_argument('--epsilon', type=float, default=1e-5, help=' VQVAE Exponential Moving Average epsilon value')
    parser.add_argument('--vqvae_ckpt', type=str, default=os.path.join('model_weights', 'conditional_vqvae_results', 'models', '5000.pth'), help='VQVAE Saved weights')
    
    # Shared between PixelCNN and VQVAE
    parser.add_argument('--embedding_dim', type=int, default=64, help='VQVAE Embedding Dimension (k)')
    
    # PixelCNN
    parser.add_argument('--pixelcnn_hidden_size_prior', type=int, default=64, help='PixelCNN Prior Hidden Size')
    parser.add_argument('--pixelcnn_num_layers', type=int, default=15, help='PixelCNN Number of Layers')
    
    args = parser.parse_args()
    print(args)
    
    train(args)