"""
Class Conditional ddim from:-
https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb#scrollTo=gmpFp4OUsPBr
"""

import os 
import argparse
import torch
from torchvision.utils import save_image
from diffusers import DDIMScheduler
from constants.labels import CLASS_NAME_TO_LABELS_MAP
from tqdm import tqdm
from models.unet.model import UNet
import uuid

def evaluate(args):
    model_weight = args.model_weight
    save_path = args.save_path
    batch_size = args.batch_size
    image_size = args.image_size
    num_iterations = args.num_iterations
    num_timesteps = args.timesteps
    num_channels = 3
    
    # Specify Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WEIGHTS_PATH = model_weight
    SAVE_DIR = save_path
    
    if not os.path.exists(os.path.join(SAVE_DIR)):
        os.makedirs(os.path.join(SAVE_DIR))
    
    # Load Model Weights
    checkpoint = torch.load(WEIGHTS_PATH)
    model = UNet(image_size).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create a Scheduler
    noise_scheduler = DDIMScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)
    
    for iteration in tqdm(range(num_iterations)):
        # Prepare X and Y
        x = torch.randn((batch_size, num_channels, image_size, image_size)).to(DEVICE)
        
        # Sampling Loop
        for index, timestep in tqdm(enumerate(noise_scheduler.timesteps)):
            # Get model pred
            with torch.no_grad():
                residual = model(x, timestep)
                
            # Update Sample with Step
            x = noise_scheduler.step(residual, timestep, x).prev_sample
        
        # Convert Tensors to Images
        for index, image in enumerate(x):
            file_name = f"{str(uuid.uuid4())}.jpg"
            save_image(image, os.path.join(SAVE_DIR, file_name), normalize=True)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--model_weight', type=str, default=os.path.join('model_weights', 'ddim_results', 'models', '25000.pth'), help='Model weight path')
    parser.add_argument('--save_path', type=str, default=os.path.join('temp', 'ddim_eval_results'), help='DEFAULT: Save generated image folder')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--num_iterations', type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    evaluate(args)
