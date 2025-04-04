# # Graph plot for Training and Validation Accuracy
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = r"D:\CodingProjects\machine_learning\Experiment_5_Latest\Visualisation_Revision_1\Cond_GatedPixelCNN+VQVAE_All_Variance.csv"
# df = pd.read_csv(file_path)

# # Identify all unique dataset names
# dataset_names = df['Dataset Name'].unique()

# # Define markers and colors to distinguish datasets
# markers = ['o', 's', 'v', '^', 'D', '*']
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# # Plotting
# plt.figure(figsize=(12, 7))

# for i, dataset in enumerate(dataset_names):
#     data = df[df['Dataset Name'] == dataset]
#     plt.plot(
#         data['Epoch'], 
#         data['Train Accuracy'], 
#         label=f'Train ({dataset})',
#         marker=markers[i % len(markers)], 
#         linestyle='--', 
#         color=colors[i % len(colors)]
#     )
#     plt.plot(
#         data['Epoch'], 
#         data['Validation Accuracy'], 
#         label=f'Validation ({dataset})',
#         marker=markers[i % len(markers)], 
#         linestyle='-', 
#         color=colors[i % len(colors)]
#     )

# # Formatting
# plt.title('GatedPixelCNN+VQVAE Ratio Datasets')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend(loc='lower right', fontsize='small')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

################### Confusion matrix generation #######################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Define your own confusion matrix values (normalized)
cm_normalized = np.array([
    [298, 2, 0],
    [1, 299, 0],
    [2, 0, 298],
])

# Define the class labels
labels = ['C.R', 'C.V', 'S.P']

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, text_kw={'fontsize': 14})  # Increase number font size

# Customize the plot
plt.title('Test (100% VQVAE)', fontsize=16)
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=45, fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
