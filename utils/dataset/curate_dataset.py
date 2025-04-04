import os
import argparse
import random
import math
from tqdm import tqdm
from PIL import Image

def save_images(images: list[str], image_size:tuple[int, int], save_path: str, class_name: str):
    if not images or len(images) < 1:
        return
    
    for image_details in tqdm(images, desc=f"Class: {class_name}"):
            image_path, image_name = image_details[0], image_details[1]
            current_image = Image.open(image_path)
            current_image = current_image.resize(image_size)
            current_image.save(os.path.join(save_path, class_name, f"{image_name[:len(image_name) - 4]}.jpg"))

def resize_images(dataset_path: str, img_height: int, img_width: int, save_path:str, ratio: float):
    for class_name in os.listdir(dataset_path):
        specified_images_path = os.path.join(dataset_path, class_name)
        specified_images_details = [(os.path.join(specified_images_path, image_name), image_name) for image_name in os.listdir(specified_images_path)]
        dirs = ['train', 'test']
        for name in dirs:
            current_save_path = os.path.join(save_path, name, class_name)
            if not os.path.exists(current_save_path):
                os.makedirs(current_save_path)
        training_dataset_cutoff = math.floor(ratio * len(specified_images_details))
        
        random.shuffle(specified_images_details)
        training_images = specified_images_details[:training_dataset_cutoff]
        test_images = specified_images_details[training_dataset_cutoff:]
        
        save_images(training_images, (img_height, img_width), os.path.join(save_path, dirs[0]), class_name)
        save_images(test_images, (img_height, img_width), os.path.join(save_path, dirs[1]), class_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    CURRENT_DIR = os.path.dirname(os.path.join('..'))
    IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
    DEFAULT_RATIO = 0.8
    parser.add_argument('--dataset_path', type=str, default=os.path.abspath(os.path.join('generated_datasets', 'curated', 'sorted_ddim_eval_dataset')), help='DEFAULT: Root dataset folder')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT)
    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH)
    parser.add_argument('--save_name', type=str, default='ddim')
    parser.add_argument('--ratio', type=float, default=DEFAULT_RATIO)
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    img_height = args.image_height
    img_width = args.image_width
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    save_path = os.path.abspath(os.path.join(CURRENT_DIR, 'datasets', f'{args.save_name}-{IMAGE_HEIGHT}x{IMAGE_WIDTH}-{DEFAULT_RATIO}-v2'))
    ratio = args.ratio
    
    resize_images(dataset_path, img_height, img_width, save_path, ratio)