import os 
import argparse
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from graph_constants import GRAPH_CONFIG, GRAPH_LEGENDS_CONFIG, DATASET_NAME_MAPPING, MODEL_NAME_MAPPING

def prepare_combined_graphs(df, save_path):
    progress_bar = tqdm(GRAPH_CONFIG)
    for graph_type in progress_bar:
        selected_graph_config = GRAPH_CONFIG[graph_type]
        selected_dataframe_keys = selected_graph_config['dataframe']
        selected_plots = selected_graph_config['plots']
        
        plt.rcParams.update({'font.size': 16})
        
        for plot in selected_plots:
            plot_config = selected_plots[plot]
            figsize = (15.7, 22.7) if len(plot_config) < 3 else (40.7, 27.7)
            fig, axs = plt.subplots(len(plot_config), 1, figsize=figsize)
            plt.subplots_adjust(top=1.5)
            for key_config in selected_dataframe_keys:
                desired_columns = selected_dataframe_keys[key_config]
                selected_columns = desired_columns + ['Epoch', 'Dataset Name', 'Model']
                graph_indicator = ['a)', 'b)', 'c)', 'd)', 'e)']
                for index, title in enumerate(plot_config):
                    datasets = plot_config[title]
                    current_df = df[df['Dataset Name'].isin(datasets)]
                    filtered_df = current_df[selected_columns]
                    for name, data in filtered_df.groupby('Dataset Name'):
                        axs[index].plot(data['Epoch'], data[desired_columns[0]], marker='o', label=f"{name} - Train")
                        axs[index].plot(data['Epoch'], data[desired_columns[1]], marker='o', label=f"{name} - Test")
                    axs[index].set(xlabel='Epoch', ylabel=key_config)
                    axs[index].annotate(graph_indicator[index], xy=(-0.05, 1.0), xycoords='axes fraction', fontsize=24)
                    axs[index].set_xlim(0, 20)
                    axs[index].set_title(title)
                    box = axs[index].get_position()
                    axs[index].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                    ncols= 3 if title == 'GatedPixelCNN + Conditional VQVAE & Unconditional VQVAE Ratio Datasets' else 2
                    axs[index].legend(ncols=ncols, loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True)
                        
            fig.subplots_adjust(bottom=0.3, wspace=0.33)
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"{graph_type} - {plot}.png"))
            plt.close()

def plot_graph(args):
    
    results_path = args.results_path
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    
    folder_paths = [(os.path.join(results_path, folder_name), folder_name) for folder_name in os.listdir(results_path)]
    
    df = pd.DataFrame()
    
    for folder_path in folder_paths:
        path, folder_name = folder_path
        metric_path = os.path.join(path, 'metrics.csv')
        metric_df = pd.read_csv(metric_path)
        
        dataset_names = []
        model_names = []
        for _ in range(len(metric_df)):
            curr_folder_name = re.sub('-1e-4', '', folder_name)
            curr_folder_name = curr_folder_name.split('-')
            dataset_name = DATASET_NAME_MAPPING['-'.join(curr_folder_name[1 : len(curr_folder_name) - 1])]
            dataset_names.append(dataset_name)
            model_name = MODEL_NAME_MAPPING[curr_folder_name[0]]
            model_names.append(model_name)
        
        metric_df['Dataset Name'] = dataset_names
        metric_df['Model'] = model_names
        df = pd.concat([df, metric_df])
    
    df.to_csv(os.path.join(save_path, 'dataframe.csv'), index=False)
    # Prepare Combined Graphs
    prepare_combined_graphs(df, save_path)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Graph')
    parser.add_argument('--results_path', type=str, default=os.path.join('results', 'resnet18_classifications_results'), help='Results')
    parser.add_argument('--save_path', type=str, default=os.path.join('visualizations', 'line_graph_with_resnet_18'))
    
    args = parser.parse_args()
    print(args)
    
    plot_graph(args)