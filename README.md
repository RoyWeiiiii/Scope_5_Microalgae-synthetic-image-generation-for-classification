# Generating synthetic dataset for microalgae using generative models: A quantitative and qualitative evaluation with downstream classification

By Jun Wei Roy Chong, Jun Rong Brian Chong, Kuan Shiong Khoo, Huong-Yong Ting, Iwamoto Koji, Zengling Ma, Pau Loke Show

Obtaining quality microalgae datasets in terms of high quality and diversity has created a significant challenge in the artificial intelligence (AI) microalgae domain. To date, the implementation of Generative AI in the image generation space has been widely explored in the medical and biological sectors. However, this approach has not been exploited in the field of microalgae biotechnology. This present research assessed various generative models such as Fast Generative Adversarial Network (FastGAN), Gated Pixel Convolutional Neural Network prior with Vector Quantised - Variational Auto Encoder (G-PixelCNN-VQVAE), and Denoising Diffusion Implicit Model (DDIM) to generate synthetic images that can capture various features of different microalgae species including Chlorella vulgaris FSP-E (C.V), Chlamydomonas reinhardtii (C.R) and Spirulina platensis (S.P). Our results showed the Frechet Inception Distance (FID) score of 32.55, 96.44 and 147.24, respectively for the conditional FastGAN, G-PixelCNN-VQVAE and DDIM. The downstream image classification task showed that training with 100% synthetic data derived from the conditional FastGAN achieved the highest validation score of 78.58%, as compared to 99.66% with the original dataset. However, when trained on the G-PixelCNN with conditional VQVAE and combined dataset, we achieved the highest validation score of 99.80%, indicating a subtle boost in performance. 

Keywords: Synthetic image generation; Microalgae; FastGAN; DDIM; VQVAE; PixelCNN

# Folder & Files descriptions
# a) Machine learning, Deep learning, & Hybrid Stacking-ensemble models

**AI_models_Final** => Contains the model configuration of SVM regressor, XGBoost regressor, CNN, and Hybrid Stacking-Ensemble model [Base models (SVM, XGBoost) & meta-regressor models (RidgeCV, LinearRegression, DecisionTree, RandomForest, SVR, XGBoost)

**SVM-XGBoost-CNN-Hybrid-EL-Model_development.xlsx** => Development of all models and with detailed explanation on python code in excel file

**SVM-XGBoost-CNN-Hybrid-EL-Model-Datasets.xlsx** => Datasets with accuracy and loss metrics derived from each model in excel file

# b) Image and data pre-processing

**Colour feature extraction & Data normalisation** =>  Contains the python script of colour feature extraction & data normalisation for training of ML models

**SVM-XGBoost-Colour_feature_extraction_Data_normalisation.xlsx** => Colour (RGB, HSL, CMYK) feature extraction and Data normalisation (min-max scaler) for ML models in excel file

# c) Datasets for CNN & Hybrid Stacking-ensemble models

**Combined_All_Batch_Days_Camera** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using digital camera capturing device under various type of variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Combined_All_Batch_Days_Smartphone** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using smartphone capturing device under various type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

# d) Datasets for SVM & XGBoost models

**Data_Colour_index_Normalised (1V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Day_Colour_index_Normalised (2V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Abs_Day_Colour_index_Normalised (3V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period),  'Abs' (absorbance), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

# e) Original & Synthetic microalgae image datasets
The google drive [https://drive.google.com/drive/folders/1QkzOmOO_flF0Y4V9MKVSmJOiIxR3vBM7?usp=sharing] contains microalgae original and synthetic image datasets for FID evaluation, t-SNE2D plot evaluation, and classification task. The image dataset is publicly available for academic and research purposes.

# Referencing and citation
If you find the generation of synthetic microalgae images based on generative models (DDIM, FastGAN, and G-PixelCNN-VQVAE) useful in your research, please consider citing: Based on the DOI: *********Not published yet***********
