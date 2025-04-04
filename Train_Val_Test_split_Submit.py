################### Working for Train, Val, test, split ######################
import os
import argparse
import random
import shutil
from datetime import datetime
import logging

# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def split_images(args):
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    RANDOM_SEED = args.random_seed

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Create subfolders
    train_folder = os.path.join(SAVE_PATH, 'train_s')
    val_folder = os.path.join(SAVE_PATH, 'val_s')
    test_folder = os.path.join(SAVE_PATH, 'test_s')
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # Collect all image filenames
    image_files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    logger.info(f"Total images: {total}")
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Helper to copy files
    def copy_files(file_list, target_folder):
        for fname in file_list:
            src = os.path.join(DATA_PATH, fname)
            dst = os.path.join(target_folder, fname)
            shutil.copy2(src, dst)

    copy_files(train_files, train_folder)
    copy_files(val_files, val_folder)
    copy_files(test_files, test_folder)

    logger.info(f"Files copied successfully into train/val/test folders.")

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='Split images into train/val/test sets')
    data_path = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_Revision_1\ratio_datasets\20%_Ori_80%_Cond_FastGAN\spirulina"
    save_path = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_Revision_1\ratio_datasets\20%_Ori_80%_Cond_FastGAN"
    args = argparse.Namespace(data_path=data_path, save_path=save_path, random_seed=42)
    
    split_images(args)

