import os
import argparse
import random
import math
from PIL import Image
from tqdm import tqdm

ORIGINAL_MAP = {
    'chlamy': 2522,
    'fsp-e': 2400,
    'spirulina': 3256
}

def ratio_images(args):
    dataset_path_a = args.dataset_path_a
    dataset_path_b = args.dataset_path_b
    dataset_path_a_name = args.dataset_path_a_name
    dataset_path_b_name = args.dataset_path_b_name
    
    ratios = [0.2, 0.4, 0.6, 0.8]
    
    for key in ORIGINAL_MAP.keys():
        curr_folder_path_a = os.path.join(dataset_path_a, key)
        curr_folder_path_b = os.path.join(dataset_path_b, key)
        imgs_a = [os.path.join(curr_folder_path_a, img_name) for img_name in os.listdir(curr_folder_path_a)]
        imgs_b = [os.path.join(curr_folder_path_b, img_name) for img_name in os.listdir(curr_folder_path_b)]
        random.shuffle(imgs_a)
        random.shuffle(imgs_b)
        
        for ratio in ratios:
            save_path = os.path.join('ratio_datasets', f"{ratio}@{dataset_path_a_name}-{round(1 - ratio, 1)}@{dataset_path_b_name}-train")
            curr_save_path = os.path.join(save_path, key)
            if not os.path.exists(curr_save_path):
                os.makedirs(curr_save_path)
            
            new_length = math.ceil(ratio * ORIGINAL_MAP[key])
            imgs_ratio_a = imgs_a[:new_length]
            imgs_ratio_b = imgs_b[new_length:ORIGINAL_MAP[key]]
            
            for img in tqdm(imgs_ratio_a, desc=f'Part A Ratio: {ratio}. Type:{key} Len A: {len(imgs_ratio_a)} Len B: {len(imgs_ratio_b)}'):
                current_image = Image.open(img)
                current_image.save(os.path.join(curr_save_path, os.path.basename(img)))
                
            for img in tqdm(imgs_ratio_b, desc=f'Part B Ratio: {round(1 - ratio, 1)}. Type:{key} Len A: {len(imgs_ratio_a)} Len B: {len(imgs_ratio_b)}'):
                current_image = Image.open(img)
                current_image.save(os.path.join(curr_save_path, os.path.basename(img)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path_a', type=str, default=os.path.join('datasets', '1024x1024-0.8-v2', 'train'))
    parser.add_argument('--dataset_path_b', type=str, default=os.path.join('generated_datasets', 'generated', 'pixelcnn_conditional_vqvae_eval_results'))
    parser.add_argument('--dataset_path_a_name', type=str, default='1024x1024-0.8-v2-train')
    parser.add_argument('--dataset_path_b_name', type=str, default='pixelcnn_cond_vqvae_eval_results')
    args = parser.parse_args()
    ratio_images(args)