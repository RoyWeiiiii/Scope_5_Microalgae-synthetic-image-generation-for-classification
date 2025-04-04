import os 
import argparse
import torch
import torch.nn.functional as F
import uuid

from tqdm import tqdm
from torchvision import utils as vutils
from models.class_conditioned_fastgan.generator import Generator
from constants.labels import CLASS_NAME_TO_LABELS_MAP

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_params(model, new_params):
    for p, new_p in zip(model.parameters(), new_params):
        p.data.copy_(new_p)
        
def resize(img, size=256):
    return F.interpolate(img, size=size)

def batch_save(images, one_hot_vectors, save_dir):
    # For labelling
    reverse_map = {v: k for k, v in CLASS_NAME_TO_LABELS_MAP.items()}
    
    for index, image in enumerate(images):
        argmax = torch.argmax(one_hot_vectors[index])
        curr_image_label = reverse_map[argmax.item()]
        save_path = os.path.join(save_dir, curr_image_label)
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
    
    # Create save directory if does not exist
    folders = [save_dir]
    for key in CLASS_NAME_TO_LABELS_MAP:
        folders.append(os.path.join(save_dir, key))
        
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Generator Model
    netG = Generator(ngf=ngf, nz=nz, nc=3, im_size=image_size, num_classes=3).to(DEVICE)
    
    # Load Weights
    checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
    # Remove prefix `module`.
    checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
    netG.load_state_dict(checkpoint['g'])
    del checkpoint
    
    with torch.no_grad():
        for _ in tqdm(range(num_epochs)):
            noise = torch.randn(batch_size, nz).to(DEVICE)
            y = torch.randint(0, 3, (batch_size,)).to(DEVICE)
            y = F.one_hot(y, num_classes=len(CLASS_NAME_TO_LABELS_MAP))
            
            generated_images = netG(noise, y)[0]
            generated_images = resize(generated_images, image_size)
            
            batch_save(generated_images, y, save_dir)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conditional FastGAN generate images'
    )
    
    parser.add_argument('--ckpt', type=str, default=os.path.join('model_weights', 'conditional_fastgan_results', 'models', '50000.pth'))
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'conditional_fastgan_eval_results'))
    
    
    args = parser.parse_args()
    generate_images(args)