import os 
import argparse
import torch
import torch.nn.functional as F
import uuid

from tqdm import tqdm
from torchvision import utils as vutils
from models.fastgan.generator import Generator

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_params(model, new_params):
    for p, new_p in zip(model.parameters(), new_params):
        p.data.copy_(new_p)
        
def resize(img, size=256):
    return F.interpolate(img, size=size)

def batch_save(images, save_dir):
    for index, image in enumerate(images):
        save_path = os.path.join(save_dir)
        file_name = f"{str(uuid.uuid4())}.jpg"
        vutils.save_image(image.add(1).mul(0.5), os.path.join(save_path, file_name))

def generate_images(args):
    
    # Argparse
    image_size = args.image_size
    ckpt = args.ckpt
    num_epochs = args.num_epochs
    save_dir = args.save_dir
    batch_size = args.batch_size
    
    # Model config
    nz=256
    ngf=64
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generator Model
    netG = Generator(ngf=ngf, nz=nz, nc=3, im_size=image_size).to(DEVICE)
    
    # Load Weights
    checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
    # Remove prefix `module`.
    checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
    netG.load_state_dict(checkpoint['g'])
    del checkpoint
    
    with torch.no_grad():
        for _ in tqdm(range(num_epochs)):
            noise = torch.randn(batch_size, nz).to(DEVICE)
            generated_images = netG(noise)[0]
            generated_images = resize(generated_images, image_size)
            
            batch_save(generated_images, save_dir)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FastGAN generate images'
    )
    
    parser.add_argument('--ckpt', type=str, default=os.path.join('model_weights', 'fastgan_results', 'models', '50000.pth'))
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp','fastgan_eval_results'))
    
    
    args = parser.parse_args()
    generate_images(args)