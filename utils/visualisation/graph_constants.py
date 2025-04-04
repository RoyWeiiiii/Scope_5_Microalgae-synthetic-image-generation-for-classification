DATASET_NAME_MAPPING = {
    '0.2@1024x1024-0.8-v2-train-0.8@conditional_ddim_eval_results-train': '20% Original Data + 80% Conditional DDIM Data',
    '0.2@1024x1024-0.8-v2-train-0.8@conditional_fastgan_eval_results-train': '20% Original Data + 80% Conditional FastGAN Data',
    '0.2@1024x1024-0.8-v2-train-0.8@conditional_vqvae_eval_results-train': '20% Original Data + 80% Conditional VQVAE Data',
    '0.2@1024x1024-0.8-v2-train-0.8@pixelcnn_cond_vqvae_eval_results-train': f'20% Original Data + 80% GatedPixelCNN + Conditional \n VQVAE Data',
    '0.2@1024x1024-0.8-v2-train-0.8@pixelcnn_vqvae_eval_results-train': f'20% Original Data + 80% GatedPixelCNN + \n VQVAE Data',
    '0.4@1024x1024-0.8-v2-train-0.6@conditional_ddim_eval_results-train': '40% Original Data + 60% Conditional DDIM Data',
    '0.4@1024x1024-0.8-v2-train-0.6@conditional_fastgan_eval_results-train': '40% Original Data + 60% Conditional FastGAN Data',
    '0.4@1024x1024-0.8-v2-train-0.6@conditional_vqvae_eval_results-train': '40% Original Data + 60% Conditional VQVAE Data',
    '0.4@1024x1024-0.8-v2-train-0.6@pixelcnn_cond_vqvae_eval_results-train': f'40% Original Data + 60% GatedPixelCNN + Conditional \n VQVAE Data',
    '0.4@1024x1024-0.8-v2-train-0.6@pixelcnn_vqvae_eval_results-train': f'40% Original Data + 60% GatedPixelCNN + \n VQVAE Data',
    '0.6@1024x1024-0.8-v2-train-0.4@conditional_ddim_eval_results-train': '60% Original Data + 40% Conditional DDIM Data',
    '0.6@1024x1024-0.8-v2-train-0.4@conditional_fastgan_eval_results-train': '60% Original Data + 40% Conditional FastGAN Data',
    '0.6@1024x1024-0.8-v2-train-0.4@conditional_vqvae_eval_results-train': '60% Original Data + 40% Conditional VQVAE Data',
    '0.6@1024x1024-0.8-v2-train-0.4@pixelcnn_cond_vqvae_eval_results-train': f'60% Original Data + 40% GatedPixelCNN + Conditional \n VQVAE Data',
    '0.6@1024x1024-0.8-v2-train-0.4@pixelcnn_vqvae_eval_results-train': f'60% Original Data + 40% GatedPixelCNN + \n VQVAE Data',
    '0.8@1024x1024-0.8-v2-train-0.2@conditional_ddim_eval_results-train': '80% Original Data + 20% Conditional DDIM Data',
    '0.8@1024x1024-0.8-v2-train-0.2@conditional_fastgan_eval_results-train': '80% Original Data + 20% Conditional FastGAN Data',
    '0.8@1024x1024-0.8-v2-train-0.2@conditional_vqvae_eval_results-train': '80% Original Data + 20% Conditional VQVAE Data',
    '0.8@1024x1024-0.8-v2-train-0.2@pixelcnn_cond_vqvae_eval_results-train': f'80% Original Data + 20% GatedPixelCNN + Conditional \n VQVAE Data',
    '0.8@1024x1024-0.8-v2-train-0.2@pixelcnn_vqvae_eval_results-train': f'80% Original Data + 20% GatedPixelCNN + \n VQVAE Data',
    '1024x1024-0.8-v2-train': '100% Original Data',
    'cond_ddim-256x256-0.8-v2-train': '100% Conditional DDIM Data',
    'cond_fastgan-256x256-0.8-v2-train': '100% Conditional FastGAN Data',
    'cond_vqvae-256x256-0.8-v2-train': '100% Conditional VQVAE Data',
    'pixelcnn_cond_vqvae-256x256-0.8-v2-train': f'100% GatedPixelCNN + Conditional \n VQVAE Data',
    'pixelcnn_vqvae-256x256-0.8-v2-train': f'100% GatedPixelCNN + \n VQVAE Data',
    'cond_ddim+1024x1024-0.8-v2-train': '100% Conditional DDIM Data + 100% Original Data',
    'cond_fastgan+1024x1024-0.8-v2-train': '100% Conditional FastGAN Data + 100% Original Data',
    'cond_vqvae+1024x1024-0.8-v2-train': '100% Conditional VQVAE Data + 100% Original Data',
    'pixelcnn_cond_vqvae+1024x1024-0.8-v2-train': f'100% GatedPixelCNN + Conditional \n VQVAE Data + 100% Original Data',
    'pixelcnn_vqvae+1024x1024-0.8-v2-train': f'100% GatedPixelCNN + \n VQVAE Data + 100% Original Data',
}

MODEL_NAME_MAPPING = {
    'resnet50': 'Resnet-50',
    'vgg16': 'VGG 16',
    'vit': 'ViT',
    'resnet18': 'Resnet-18'
}

GRAPH_LEGENDS_CONFIG = {
    'DDIM Ratio Datasets': (0.75, 0.15),
    'VQVAE Ratio Datasets': (0.75, 0.15),
    'GatedPixelCNN + Conditional VQVAE Ratio Datasets': (0.75, 0),
    'FastGAN Ratio Datasets': (0.75, 0.15),
    '100% Original vs 100% Synthetic': (0.70, 0.21),
    'Combined Datasets': (0.73, 0.252)
}

GRAPH_CONFIG = {
    'Accuracy Per Epoch': {
        'models': ['Resnet-18'],
        'dataframe': {
            'Accuracy': ['Training Accuracy', 'Testing Accuracy'],
        },
        'plots': {
            '100% Original vs 100% Synthetic + Combined Datasets': {
                '100% Original vs 100% Synthetic': [
                    '100% Original Data',
                    '100% Conditional DDIM Data',
                    '100% Conditional FastGAN Data',
                    f'100% GatedPixelCNN + \n VQVAE Data',
                    f'100% GatedPixelCNN + Conditional \n VQVAE Data'
                ],
                'Combined Datasets': [
                    '100% Original Data',
                    '100% Conditional DDIM Data + 100% Original Data',
                    '100% Conditional FastGAN Data + 100% Original Data',
                    '100% Conditional VQVAE Data + 100% Original Data',
                    f'100% GatedPixelCNN + \n VQVAE Data + 100% Original Data',
                    f'100% GatedPixelCNN + Conditional \n VQVAE Data + 100% Original Data',
                ],
            },
            'Ratio Datasets Part 1': {
                'DDIM Ratio Datasets': [
                    '100% Original Data',
                    '20% Original Data + 80% Conditional DDIM Data',
                    '40% Original Data + 60% Conditional DDIM Data',
                    '60% Original Data + 40% Conditional DDIM Data',
                    '80% Original Data + 20% Conditional DDIM Data',
                    '100% Conditional DDIM Data',
                ],
                'GatedPixelCNN + Conditional VQVAE Ratio Datasets': [
                    '100% Original Data',
                    f'20% Original Data + 80% GatedPixelCNN + Conditional \n VQVAE Data',
                    f'40% Original Data + 60% GatedPixelCNN + Conditional \n VQVAE Data',
                    f'60% Original Data + 40% GatedPixelCNN + Conditional \n VQVAE Data',
                    f'80% Original Data + 20% GatedPixelCNN + Conditional \n VQVAE Data',
                    f'100% GatedPixelCNN + Conditional \n VQVAE Data'
                ],
            },
            'Ratio Datasets Part 2': {
                'GatedPixelCNN + VQVAE Ratio Datasets': [
                    '100% Original Data',
                    f'20% Original Data + 80% GatedPixelCNN + \n VQVAE Data',
                    f'40% Original Data + 60% GatedPixelCNN + \n VQVAE Data',
                    f'60% Original Data + 40% GatedPixelCNN + \n VQVAE Data',
                    f'80% Original Data + 20% GatedPixelCNN + \n VQVAE Data',
                    f'100% GatedPixelCNN + \n VQVAE Data',
                ],
                'FastGAN Ratio Datasets': [
                    '100% Original Data',
                    '20% Original Data + 80% Conditional FastGAN Data',
                    '40% Original Data + 60% Conditional FastGAN Data',
                    '60% Original Data + 40% Conditional FastGAN Data',
                    '80% Original Data + 20% Conditional FastGAN Data',
                    '100% Conditional FastGAN Data',
                ],
            }
        }
    },
}

# GRAPH_CONFIG = {
#     'Loss Per Epoch': {
#         # 'models': ['Resnet-50', 'VGG 16', 'ViT'],
#         'models': ['Resnet-18'],
#         'dataframe': {
#             'Training and Testing Loss': ['Training Loss', 'Testing Loss'],
#         },
#         'datasets': {
#             '100% Original vs 100% Synthetic': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#                 '100% Conditional FastGAN Data',
#                 '100% Conditional VQVAE Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data'
#             ],
#             'DDIM Ratio Datasets': [
#                 '20% Original Data + 80% Conditional DDIM Data',
#                 '40% Original Data + 60% Conditional DDIM Data',
#                 '60% Original Data + 40% Conditional DDIM Data',
#                 '80% Original Data + 20% Conditional DDIM Data',
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#             ],
#             'VQVAE Ratio Datasets': [
#                 '20% Original Data + 80% Conditional VQVAE Data',
#                 '40% Original Data + 60% Conditional VQVAE Data',
#                 '60% Original Data + 40% Conditional VQVAE Data',
#                 '80% Original Data + 20% Conditional VQVAE Data',
#                 '100% Original Data',
#                 '100% Conditional VQVAE Data',
#             ],
#             'GatedPixelCNN + Conditional VQVAE Ratio Datasets': [
#                 f'20% Original Data + 80% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'40% Original Data + 60% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'60% Original Data + 40% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'80% Original Data + 20% GatedPixelCNN + Conditional \n VQVAE Data',
#                 '100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data',
#             ],
#             'FastGAN Ratio Datasets': [
#                 '20% Original Data + 80% Conditional FastGAN Data',
#                 '40% Original Data + 60% Conditional FastGAN Data',
#                 '60% Original Data + 40% Conditional FastGAN Data',
#                 '80% Original Data + 20% Conditional FastGAN Data',
#                 '100% Original Data',
#                 '100% Conditional FastGAN Data',
#             ],
#             'Combined Datasets': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data + 100% Original Data',
#                 '100% Conditional FastGAN Data + 100% Original Data',
#                 '100% Conditional VQVAE Data + 100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data + 100% Original Data',
#             ]
#         }
#     },
#     'Accuracy Per Epoch': {
#         # 'models': ['Resnet-50', 'VGG 16', 'ViT'],
#         'models': ['Resnet-18'],
#         'dataframe': {
#             'Training and Testing Accuracy': ['Training Accuracy', 'Testing Accuracy'],
#         },
#         'datasets': {
#             '100% Original vs 100% Synthetic': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#                 '100% Conditional FastGAN Data',
#                 '100% Conditional VQVAE Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data'
#             ],
#             'DDIM Ratio Datasets': [
#                 '20% Original Data + 80% Conditional DDIM Data',
#                 '40% Original Data + 60% Conditional DDIM Data',
#                 '60% Original Data + 40% Conditional DDIM Data',
#                 '80% Original Data + 20% Conditional DDIM Data',
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#             ],
#             'GatedPixelCNN + Conditional VQVAE Ratio Datasets': [
#                 f'20% Original Data + 80% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'40% Original Data + 60% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'60% Original Data + 40% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'80% Original Data + 20% GatedPixelCNN + Conditional \n VQVAE Data',
#                 '100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data',
#             ],
#             'VQVAE Ratio Datasets': [
#                 '20% Original Data + 80% Conditional VQVAE Data',
#                 '40% Original Data + 60% Conditional VQVAE Data',
#                 '60% Original Data + 40% Conditional VQVAE Data',
#                 '80% Original Data + 20% Conditional VQVAE Data',
#                 '100% Original Data',
#                 '100% Conditional VQVAE Data',
#             ],
#             'FastGAN Ratio Datasets': [
#                 '20% Original Data + 80% Conditional FastGAN Data',
#                 '40% Original Data + 60% Conditional FastGAN Data',
#                 '60% Original Data + 40% Conditional FastGAN Data',
#                 '80% Original Data + 20% Conditional FastGAN Data',
#                 '100% Original Data',
#                 '100% Conditional FastGAN Data',
#             ],
#             'Combined Datasets': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data + 100% Original Data',
#                 '100% Conditional FastGAN Data + 100% Original Data',
#                 '100% Conditional VQVAE Data + 100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data + 100% Original Data',
#             ]
#         }
#     },
#     'F1 Score Per Epoch': {
#         # 'models': ['Resnet-50', 'VGG 16', 'ViT'],
#         'models': ['Resnet-18'],
#         'dataframe': {
#             'Training and Testing F1 Score': ['Training F1 Score', 'Testing F1 Score'],
#         },
#         'datasets': {
#             '100% Original vs 100% Synthetic': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#                 '100% Conditional FastGAN Data',
#                 '100% Conditional VQVAE Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data'
#             ],
#             'GatedPixelCNN + Conditional VQVAE Ratio Datasets': [
#                 f'20% Original Data + 80% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'40% Original Data + 60% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'60% Original Data + 40% GatedPixelCNN + Conditional \n VQVAE Data',
#                 f'80% Original Data + 20% GatedPixelCNN + Conditional \n VQVAE Data',
#                 '100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data',
#             ],
#             'DDIM Ratio Datasets': [
#                 '20% Original Data + 80% Conditional DDIM Data',
#                 '40% Original Data + 60% Conditional DDIM Data',
#                 '60% Original Data + 40% Conditional DDIM Data',
#                 '80% Original Data + 20% Conditional DDIM Data',
#                 '100% Original Data',
#                 '100% Conditional DDIM Data',
#             ],
#             'VQVAE Ratio Datasets': [
#                 '20% Original Data + 80% Conditional VQVAE Data',
#                 '40% Original Data + 60% Conditional VQVAE Data',
#                 '60% Original Data + 40% Conditional VQVAE Data',
#                 '80% Original Data + 20% Conditional VQVAE Data',
#                 '100% Original Data',
#                 '100% Conditional VQVAE Data',
#             ],
#             'FastGAN Ratio Datasets': [
#                 '20% Original Data + 80% Conditional FastGAN Data',
#                 '40% Original Data + 60% Conditional FastGAN Data',
#                 '60% Original Data + 40% Conditional FastGAN Data',
#                 '80% Original Data + 20% Conditional FastGAN Data',
#                 '100% Original Data',
#                 '100% Conditional FastGAN Data',
#             ],
#             'Combined Datasets': [
#                 '100% Original Data',
#                 '100% Conditional DDIM Data + 100% Original Data',
#                 '100% Conditional FastGAN Data + 100% Original Data',
#                 '100% Conditional VQVAE Data + 100% Original Data',
#                 f'100% GatedPixelCNN + Conditional \n VQVAE Data + 100% Original Data',
#             ]
#         }
#     }
# }