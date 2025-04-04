# Generating synthetic dataset for microalgae using generative models: A quantitative and qualitative evaluation with downstream classification

By Jun Wei Roy Chong, Jun Rong Brian Chong, Kuan Shiong Khoo, Huong-Yong Ting, Iwamoto Koji, Zengling Ma, Pau Loke Show

******************* Not available at the moment ***********************
![Overall_Pipeline_V1](https://github.com/user-attachments/assets/f8a81909-f7a6-4b21-a1e6-6d919d3b1e89)



Keywords: ************* Not available at the moment ********************

# Folder & Files descriptions
# a) CNN model development for classification task & Generative models for synthetic image synthesis

**Model_classification_CNN_Pytorch_Final** => Python script for the development of a custom CNN model to perform classification of original & synthetic image datasets

**Generative_models_Evaluation_Final** => Python script to generate synthetic microalgae images based on unconditional & conditional DDIM, FastGAN, and G-PixelCNN-VQVAE

**Generative_models_Training_Final** => Python script to train unconditional & conditional generative models (DDIM, FastGAN, and G-PixelCNN-VQVAE)

# b) Image and data pre-processing

**Image_preprocessing_resize_Submit** =>  Python script for image resizing

**Train_Val_Test_split_Submit** => Python script to split images into training, validation, and testing folders

# c) Task evaluation and visualisation

**Compute_FID_score_Pytorch_Submit** => Python script to compute FID scores between original and synthetic images 

**t-SNE_Resnet-50_method-wise_Submit** => Python script to generate t-SNE 2D plot from original and synthetic images based on Resnet-50

**Train_Val_Graph_Confusion_matrix_Submit** => Python script to generate training and validation accuracy and loss learning curves along with confusion matrix

# c) Original & Synthetic microalgae image datasets
The google drive [https://drive.google.com/drive/folders/1QkzOmOO_flF0Y4V9MKVSmJOiIxR3vBM7?usp=sharing] contains microalgae original (1024 x 1024 pixels) and synthetic image datasets (256 x 256 pixels). Original and synthetic microalgae datasets cater to FID evaluation, t-SNE2D plot evaluation, and classification tasks. All training, validation, and testing results are also available here. The image dataset is publicly available for academic and research purposes.

# Referencing and citation
If you find the generation of synthetic microalgae images based on generative models (DDIM, FastGAN, and G-PixelCNN-VQVAE) useful in your research, please consider citing: Based on the DOI: *********Not published yet***********
