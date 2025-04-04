import os
import argparse
import random
from tqdm import tqdm
from PIL import Image

def reduce_images(dataset_path: str, img_height: int, img_width: int, save_path:str):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    random.shuffle(img_paths)
    img_paths = img_paths[:7070]
    
    for img_path in tqdm(img_paths):
        curr_img = Image.open(img_path)
        curr_img = curr_img.resize((img_height, img_width))
        curr_img.save(os.path.join(save_path, os.path.basename(img_path)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    CURRENT_DIR = os.path.dirname(os.path.join('..'))
    IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
    parser.add_argument('--dataset_path', type=str, default=os.path.join('fid_datasets', 'modified', 'original-v2-256x256-v2'), help='DEFAULT: Root dataset folder')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT)
    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH)
    parser.add_argument('--save_name', type=str, default='reduced-original')
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    img_height = args.image_height
    img_width = args.image_width
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    save_path = os.path.abspath(os.path.join(CURRENT_DIR, 'fid_datasets', 'modified', f'{args.save_name}-{IMAGE_HEIGHT}x{IMAGE_WIDTH}-v2'))
    
    reduce_images(dataset_path, img_height, img_width, save_path)