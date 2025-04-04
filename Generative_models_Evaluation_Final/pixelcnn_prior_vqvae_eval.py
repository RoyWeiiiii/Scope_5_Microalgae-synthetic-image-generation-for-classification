import argparse
import os
import torch
import uuid
from torchvision import utils as vutils
from tqdm.auto import tqdm

from models.vqvae.model import VQVAE
from models.gated_pixelcnn.model import GatedPixelCNN
from constants.labels import CLASS_NAME_TO_LABELS_MAP

# Torch Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(args):
    # Argparse Values
    total_iterations = args.total_iterations
    batch_size = args.batch_size
    pixelcnn_prior_dims = args.pixelcnn_prior_dims
    save_dir = os.path.join(args.save_dir)
    
    # VQVAE Config
    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    beta = args.beta
    decay = args.decay
    epsilon = args.epsilon
    vqvae_checkpoint = args.vqvae_ckpt
    
    LABEL_TO_CLASS = {v: k for k, v in CLASS_NAME_TO_LABELS_MAP.items()}
    
    # PixelCNN
    pixelcnn_hidden_size_prior = args.pixelcnn_hidden_size_prior
    pixelcnn_num_layers = args.pixelcnn_num_layers
    pixelcnn_checkpoint = args.ckpt
    
    num_classes = len(CLASS_NAME_TO_LABELS_MAP)
    current_iteration = 0
    
    # Create save directory if does not exist
    for key in CLASS_NAME_TO_LABELS_MAP.keys():
        if not os.path.exists(os.path.join(save_dir, key)):
            os.makedirs(os.path.join(save_dir, key))
    
    # Prepare model
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              beta, decay, epsilon).to(DEVICE)
    
    # Load VQVAE weights
    vqvae_ckpt = torch.load(vqvae_checkpoint)
    model.load_state_dict(vqvae_ckpt['model'])
    model.eval()
    
    # GatedPixelCNN
    pixelcnn_prior = GatedPixelCNN(num_embeddings, pixelcnn_hidden_size_prior, pixelcnn_num_layers, num_classes).to(DEVICE)
    pixelcnn_ckpt = torch.load(pixelcnn_checkpoint)
    pixelcnn_prior.load_state_dict(pixelcnn_ckpt['model'])
    pixelcnn_prior.eval()
    
    progress_bar = tqdm(range(current_iteration, total_iterations + 1))
    for _ in progress_bar:
        with torch.no_grad():
            labels = torch.Tensor([label % num_classes for label in range(batch_size)]).long().to(DEVICE)
            priors = pixelcnn_prior.generate(labels, shape=pixelcnn_prior_dims, batch_size=batch_size)
            
            images = model.decode(priors)
            for index, image in tqdm(enumerate(images), desc='Saving Images'):
                curr_label = labels[index].cpu().numpy().item()
                class_name = LABEL_TO_CLASS[curr_label]
                file_name = f"{str(uuid.uuid4())}.jpg"
                vutils.save_image(image, os.path.join(save_dir, class_name, file_name), normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQVAE')
    parser.add_argument('--total_iterations', type=int, default=1, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--save_dir', type=str, default=os.path.join('temp', 'pixelcnn_vqvae_eval_results'), help='output results location')
    parser.add_argument('--ckpt', type=str, default=os.path.join('model_weights', 'pixelcnn_vqvae', 'models', '20000.pth'), help='checkpoint weight path if have one')
    
    #VQVAE
    parser.add_argument('--num_hiddens', type=int, default=128, help='VQVAE Hidden Layer Size')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='VQVAE Residual Hidden Size')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='VQVAE Number of Residual Layers')
    parser.add_argument('--num_embeddings', type=int, default=512, help='VQVAE Embedding Dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='VQVAE Commitment Cost or Beta')
    parser.add_argument('--decay', type=float, default=0, help='VQVAE Exponential Moving Average decay value')
    parser.add_argument('--epsilon', type=float, default=1e-5, help=' VQVAE Exponential Moving Average epsilon value')
    parser.add_argument('--vqvae_ckpt', type=str, default=os.path.join('model_weights', 'vqvae_results', 'models', '5000.pth'), help='VQVAE Saved weights')
    
    # Shared between PixelCNN and VQVAE
    parser.add_argument('--embedding_dim', type=int, default=64, help='VQVAE Embedding Dimension (k)')
    
    # PixelCNN
    parser.add_argument('--pixelcnn_hidden_size_prior', type=int, default=64, help='PixelCNN Prior Hidden Size')
    parser.add_argument('--pixelcnn_num_layers', type=int, default=15, help='PixelCNN Number of Layers')
    parser.add_argument('--pixelcnn_prior_dims', type=int, default=(64, 64), help='PixelCNN Prior Dims (N x 4)')
    
    args = parser.parse_args()
    print(args)
    
    evaluate(args)