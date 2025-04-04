import argparse
import os
import torch
import torch.nn as nn 
import torch.optim as optim
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm
from diffusers import DDIMScheduler

from dataset_loader.microalgae_dataset import MicroalgaeDataset
from constants.labels import CLASS_NAME_TO_LABELS_MAP
from utils import DatasetSampler
from models.class_conditioned_unet.model import ClassConditionedUNet

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, x_shape, num_classes, noise_scheduler, iter, save_dir):
    # Prepare X and Y
    x = torch.randn(x_shape).to(DEVICE)
    y = torch.randint(0, num_classes, (x_shape[0],)).to(DEVICE)
    
    # Sampling Loop
    for _, timestep in tqdm(enumerate(noise_scheduler.timesteps)):
        # Get model pred
        with torch.no_grad():
            residual = model(x, timestep, y)
            
        # Update Sample with Step
        x = noise_scheduler.step(residual, timestep, x).prev_sample
    
    reverse_map = {v: k for k, v in CLASS_NAME_TO_LABELS_MAP.items()}
    label_names = []
    for _, label in enumerate(y):
        curr_label = label.cpu().numpy().item()
        label_names.append(reverse_map[curr_label])
    
    labels = ' '.join(label_names)
    
    vutils.save_image(x, os.path.join(save_dir, f"iter_{iter}-class_{labels}.png"), normalize=True)


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
    inference_timesteps = args.inference_timesteps
    training_timesteps = args.training_timesteps
    
    # Set seed
    torch.manual_seed(seed)
    
    # Model Config
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
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    
    # Prepare Dataset
    train_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path), transforms=transforms_list)
    train_sampler = DatasetSampler(train_dataset)
    train_dataloader = iter(DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=dataloader_workers))
    
    # Declare model
    model = ClassConditionedUNet(image_size).to(DEVICE)
    
    # Create a Scheduler
    noise_scheduler = DDIMScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=inference_timesteps)
    
    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # Loss Function
    criterion = nn.MSELoss()
    
    # Load checkpoint if exists
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        current_iteration = ckpt['iteration']
        del ckpt
    
    # Metrics
    loss_values = []
    
    # X Shape
    x_shape = (batch_size, *train_dataset[0][0].shape) # BCHW
    
    model.train()
    print(current_iteration, total_iterations + 1)
    progress_bar = tqdm(range(current_iteration, total_iterations + 1))
    for iteration in progress_bar:
        # Retrieve Training Data
        images, labels = next(train_dataloader)
        x, y = images.to(DEVICE), labels.to(DEVICE)
        
        # Prepare noise, timesteps
        noise = torch.randn_like(x).to(DEVICE)
        timesteps = torch.randint(0, training_timesteps, (x.shape[0],)).long().to(DEVICE) # torch.randint will add 1 to the max range value
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        
        # Get Predictions
        pred = model(noisy_x, timesteps, y)
            
        # Calculate Loss
        loss = criterion(pred, noise)
            
        # Backprop and Update Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store Metrics
        loss_values.append([iteration, loss.item()])
        
        # Update Progress Bar
        progress_bar.set_description(f"Loss per Iteration: {loss.item():05f}")
        
        if iteration % save_interval == 0 or iteration == total_iterations:
            # Sample Images
            evaluate(model, x_shape, num_classes, noise_scheduler, iteration, saved_image_folder)

        if iteration % save_interval == 0 or iteration == total_iterations:
            # Save Weights
            model_weights = model.state_dict()
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
    parser = argparse.ArgumentParser(description='Conditional DDIM')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', '1024x1024-0.8-v2', 'train'), help='Microalgae Training Dataset location')
    parser.add_argument('--total_iterations', type=int, default=25000, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--training_timesteps', type=int, default=999)
    parser.add_argument('--inference_timesteps', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=256, help='Image Resolution Size')
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'conditional_ddim_train_results'), help='output results location')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of iterations to save model')
    parser.add_argument('--seed', type=int, default=42, help='PyTorch Seed')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    
    args = parser.parse_args()
    print(args)
    
    train(args)