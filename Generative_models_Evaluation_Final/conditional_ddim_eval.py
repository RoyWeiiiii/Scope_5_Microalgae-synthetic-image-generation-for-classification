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
from models.class_conditioned_unet.model import ClassConditionedUNet

def evaluate(args):
    model_weight = args.model_weight
    save_path = args.save_path
    batch_size = args.batch_size
    image_size = args.image_size
    num_iterations = args.num_iterations
    num_timesteps = args.timesteps
    num_channels = 3
    num_classes = len(CLASS_NAME_TO_LABELS_MAP)
    
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
    model = ClassConditionedUNet(image_size).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create a Scheduler
    noise_scheduler = DDIMScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)
    
    for iteration in tqdm(range(num_iterations)):
        # Prepare X and Y
        x = torch.randn((batch_size, num_channels, image_size, image_size)).to(DEVICE)
        y = torch.Tensor([label % num_classes for label in range(batch_size)]).int().to(DEVICE)
        
        # Sampling Loop
        for index, timestep in tqdm(enumerate(noise_scheduler.timesteps)):
            # Get model pred
            with torch.no_grad():
                residual = model(x, timestep, y)
                
            # Update Sample with Step
            x = noise_scheduler.step(residual, timestep, x).prev_sample
        
        # Convert Tensors to Images
        for index, image in enumerate(x):
            curr_label = y[index].cpu().numpy().item()
            class_name = LABEL_TO_CLASS[curr_label]
            save_image(image, os.path.join(SAVE_DIR, class_name, f"{(iteration * batch_size) + index}.png"), normalize=True)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--model_weight', type=str, default=os.path.join('temp', 'model_weights', 'conditional_ddim_train_results', 'models', '25000.pth'), help='Model weight path')
    parser.add_argument('--save_path', type=str, default=os.path.join('temp', 'conditional_ddim_eval_results'), help='DEFAULT: Save generated image folder')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--num_iterations', type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    evaluate(args)
