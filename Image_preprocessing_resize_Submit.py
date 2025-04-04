############################# Image resized per class ##############################
import os
from PIL import Image

def resize_class_images(source_class_dir, target_class_dir, size=(1024, 1024)):
    """
    Resize all images in a specific class folder and save them to a new directory.

    Args:
        source_class_dir (str): Path to the original class folder (e.g., .../chlamy).
        target_class_dir (str): Path to save resized images (e.g., .../resized/chlamy).
        size (tuple): Target image size, e.g., (256, 256).
    """
    os.makedirs(target_class_dir, exist_ok=True)
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(source_class_dir):
        if filename.lower().endswith(supported_formats):
            src_path = os.path.join(source_class_dir, filename)
            dst_path = os.path.join(target_class_dir, filename)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(size, Image.Resampling.LANCZOS)  # Fixed here
                img.save(dst_path)
                print(f"✅ Saved resized image: {dst_path}")
            except Exception as e:
                print(f"❌ Failed to process {src_path}: {e}")

# 🔧 Example usage
resize_class_images(
    source_class_dir=r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_Revision_1\original_datasets\spirulina",
    target_class_dir=r"D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_256 x 256\original_datasets\spirulina",
    size=(256, 256)
)
