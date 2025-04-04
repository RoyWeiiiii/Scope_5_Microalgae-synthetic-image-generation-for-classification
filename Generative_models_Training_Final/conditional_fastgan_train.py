import argparse
import random
import os
import lpips
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from constants.labels import CLASS_NAME_TO_LABELS_MAP
from models.class_conditioned_fastgan.discriminator import Discriminator
from models.class_conditioned_fastgan.generator import Generator
from models.class_conditioned_fastgan.blocks.helper import weights_init
from dataset_loader.microalgae_dataset import MicroalgaeDataset
from utils import DiffAugment, DatasetSampler, copy_G_params, load_params

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FastGan usage
POLICY = 'color,translation'
PERCEPT = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True) # Will repeatedly initialize so we will declare globally

#torch.backends.cudnn.benchmark = True
def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, y, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, y, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            PERCEPT( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            PERCEPT( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            PERCEPT( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label, y)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

# Training Function
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
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002 # Learning Rate == 2e-4
    nbeta1 = 0.5
    
    # Set seed
    torch.manual_seed(seed)
    
    num_classes = len(CLASS_NAME_TO_LABELS_MAP)
    current_iteration = 0
    saved_image_folder = os.path.join(save_dir, 'images')
    saved_image_rec_folder = os.path.join(save_dir, 'rec_images')
    saved_model_folder = os.path.join(save_dir, 'models')
    
    # Create save directory if does not exist
    folders = [save_dir, saved_image_folder, saved_image_rec_folder, saved_model_folder]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    # Transforms
    transforms_list = [
        transforms.Resize((int(image_size),int(image_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    # Prepare Dataset
    train_dataset = MicroalgaeDataset(dataset_path=os.path.join(dataset_path), transforms=transforms_list)
    train_sampler = DatasetSampler(train_dataset)
    train_dataloader = iter(DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=dataloader_workers))
    
    # Prepare Generator Model
    netG = Generator(ngf=ngf, nz=nz, nc=3, im_size=image_size, num_classes=3)
    netG.apply(weights_init)
    netG.to(DEVICE)
    
    # Prepare Discriminator Model
    netD = Discriminator(ndf=ndf, nc=3, im_size=image_size, num_classes=3)
    netD.apply(weights_init)
    netD.to(DEVICE)
    
    # Copy Generator Params
    avg_param_G = copy_G_params(netG)
    
    # Prepare Generator and Discriminator optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # Load checkpoint if exists
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    
    # Metrics
    loss_values = []
    
    netG.train()
    netD.train()
    progress_bar = tqdm(range(current_iteration, total_iterations + 1))
    for iteration in progress_bar:
        # Retrieve Training Data
        real_images, real_labels = next(train_dataloader)
        real_images, real_labels = real_images.to(DEVICE), real_labels.to(DEVICE)
        
        # Prepare Conditioning Labels
        real_labels_netG = F.one_hot(real_labels, num_classes) # netG's y
        real_labels_netD = real_labels_netG.view((-1, 3, 1, 1)).repeat(1, 1, image_size, image_size) # netD's y
        
        # Store details
        current_batch_size = real_images.size(0)
        
        # Prepare noise
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(DEVICE)
        fixed_noise = torch.FloatTensor(current_batch_size, nz).normal_(0, 1).to(DEVICE)
        
        # Generate Fake Images
        fake_images = netG(noise, real_labels_netG)
        
        # Prepare Fake and Real Augment Images
        real_images = DiffAugment(real_images, policy=POLICY)
        fake_images = [DiffAugment(fake, policy=POLICY) for fake in fake_images]
        
        # Train Discriminator
        netD.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_images, real_labels_netD, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], real_labels_netD, label="fake")
        optimizerD.step()
        
        # Train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake", real_labels_netD)
        err_g = -pred_g.mean()
        err_g.backward()
        optimizerG.step()
        
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)
            
        progress_bar.set_description("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))
        loss_values.append([iteration, err_dr, -err_g.item()])
            
        if iteration % save_interval == 0 or iteration == total_iterations:
            # Sample Images
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise, real_labels_netG)[0].add(1).mul(0.5), os.path.join(saved_image_folder, f'{iteration}.jpg'), nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_images, 128), 
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), os.path.join(saved_image_rec_folder, f'rec_{iteration}.jpg'))
            load_params(netG, backup_para)
            
        if iteration % save_interval == 0 or iteration == total_iterations:
            # Save Model Weights
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, os.path.join(saved_model_folder, f'{iteration}.pth'))
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, os.path.join(saved_model_folder, f'{iteration}.pth'))
            # Save loss
            df = pd.DataFrame(loss_values, columns=['Iteration', 'Discriminator Loss', 'Generator Loss'])
            df.to_csv(os.path.join(save_dir, 'loss_per_iteration.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional-FastGAN')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('datasets', '1024x1024-0.8-v2', 'train'), help='Microalgae Training Dataset location')
    parser.add_argument('--total_iterations', type=int, default=50000, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--image_size', type=int, default=1024, help='Image Resolution Size')
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'conditional_fastgan_train_results'), help='output results location')
    parser.add_argument('--save_interval', type=int, default=10000, help='number of iterations to save model')
    parser.add_argument('--seed', type=int, default=42, help='PyTorch Seed')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    
    args = parser.parse_args()
    print(args)
    
    train(args)