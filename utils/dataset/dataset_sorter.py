import os
import argparse
import random
from PIL import Image

# Original Train Dataset Count
ORIGINAL_MAP = {
    'chlamy': 3153,
    'fsp-e': 3000,
    'spirulina': 4070
}

def sort_images(args):
    data_path = args.dataset_path
    save_path = args.save_path
    
    for key in ORIGINAL_MAP.keys():
        curr_folder_path = os.path.join(data_path, key)
        imgs = [os.path.join(curr_folder_path, img_name) for img_name in os.listdir(curr_folder_path)]
        random.shuffle(imgs)
        imgs = imgs[:ORIGINAL_MAP[key]]
        curr_save_path = os.path.join(save_path, key)
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path)
        
        for img in imgs:
            current_image = Image.open(img)
            current_image.save(os.path.join(curr_save_path, os.path.basename(img)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.path.join('generated_datasets', 'sorted', 'sorted_ddim_eval_images'))
    parser.add_argument('--save_path', type=str, default=os.path.join('generated_datasets', 'curated', 'sorted_ddim_eval_dataset'))
    
    args = parser.parse_args()
    sort_images(args)