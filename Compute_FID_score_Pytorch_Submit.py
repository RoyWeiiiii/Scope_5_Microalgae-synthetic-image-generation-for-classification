###################### Compute FID score based on https://github.com/vict0rsch/pytorch-fid-wrapper ################
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_fid_wrapper as pfw
from tqdm.auto import tqdm

def load_images_from_imagefolder(root_folder, image_size=(256, 256), batch_size=64): # Originally was 299 x 299 pixels
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=root_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_images = []
    for batch, _ in tqdm(loader, desc=f"🔄 Loading from: {os.path.basename(root_folder)}"):
        all_images.append(batch)

    return torch.cat(all_images, dim=0)

# 🔧 Paths
base_path = "D:\CodingProjects\machine_learning\Experiment_5_Latest\datasets_256x256_FID"
real_path = os.path.join(base_path, "original_datasets")
gen_base = os.path.join(base_path, "generated_datasets")

models = [
    "cond_ddim", "cond_fastgan", "ddim",
    "fastgan", "pixelcnn_cond_vqvae", "pixelcnn_vqvae"
]

# ✅ Load real images once
print("📥 Loading real images...")
real_images = load_images_from_imagefolder(real_path)

# ✅ Compute reference statistics
pfw.set_config(device="cuda")
real_m, real_s = pfw.get_stats(real_images)

# ✅ Loop over models with FID computation progress
results = []
for model in tqdm(models, desc="📊 Computing FID per model"):
    model_path = os.path.join(gen_base, model)
    print(f"\n📥 Loading generated images from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Missing: {model_path} — skipping")
        continue

    fake_images = load_images_from_imagefolder(model_path)
    if fake_images.size(0) == 0:
        print(f"⚠️ No images found in {model_path} — skipping")
        continue

    fid_score = pfw.fid(fake_images, real_m=real_m, real_s=real_s)
    print(f"✅ FID for {model}: {fid_score:.2f}")
    results.append((model, fid_score))

# ✅ Final report
print("\n📈 Final FID Results:")
for model, score in results:
    print(f"{model:25s} : {score:.2f}")
