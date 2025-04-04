import os
import argparse
import random
from PIL import Image
from tqdm import tqdm

def append_extension(args):
    
    dataset_paths = args.dataset_paths
    save_path = args.save_path
    
    for folder in os.listdir(dataset_paths):
        current_save_path = os.path.join(save_path, folder)
        if not os.path.exists(current_save_path):
            os.makedirs(current_save_path)
    
    for folder in os.listdir(dataset_paths):
        current_path = os.path.join(dataset_paths, folder)
        for image_name in tqdm(os.listdir(current_path)):
            image_path = os.path.join(current_path, image_name)
            current_image = Image.open(image_path)
            new_image_name = image_name.split('.')[0] + '.jpeg'
            current_image.save(os.path.join(save_path, folder, new_image_name))
            current_image.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_paths', type=str, default=os.path.join('fid_datasets', 'original'))
    parser.add_argument('--save_path', type=str, default=os.path.join('fid_datasets', 'modified'))
    
    args = parser.parse_args()
    append_extension(args)