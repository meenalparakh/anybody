import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import seaborn as sns
import matplotlib.colors as mpl
import plotly.io as pio

pio.kaleido.scope.mathjax = None

from anybody.utils.path_utils import get_figures_dir

def seaborn_color_to_rgb_string(seaborn_color, dark: float=0):
    """
    Converts a seaborn color specification to an 'rgb(r,g,b)' string.

    Args:
        seaborn_color: A valid seaborn color specifier (e.g., a name from
                       a palette, an RGB tuple, an HSL tuple, etc.).

    Returns:
        str: An 'rgb(r,g,b)' string representation of the color.
             Returns None if the conversion fails.
    """
    try:
        # seaborn uses matplotlib's color handling under the hood
        rgb_float = mpl.to_rgb(seaborn_color)
        # Convert float RGB (0-1) to integer RGB (0-255)
        if dark:
            rgb_float = [c * dark for c in rgb_float]
        
        rgb_int = tuple(int(255 * c) for c in rgb_float)
        return f'rgb({rgb_int[0]},{rgb_int[1]},{rgb_int[2]})'
    except ValueError:
        print(f"Error: Could not convert '{seaborn_color}' to RGB.")
        return None


def get_ft_df(df, run_dict, other_rows=[], ft="ft10"):
    zs_metrics = run_dict["zs_metrics"][:1]
    
    # modified_data = other_rows    # contain other rows.
    modified_data = []
    
    for idx, zs_name in enumerate(run_dict["zs_names"]):
        if zs_name not in ["Mlp", "Tf"]:
            continue

        row = {"Method": f"{zs_name}-{ft}"}

        for metric in zs_metrics:
            zs_run_name = run_dict["zs"][idx][:-5]  # remove the last "_eval" suffix
            # search for all names in df which starts with f"{zs_run_name}_ft_" and contain "eval" 
            ft_run_name = [name for name in df.index if name.startswith(f"{zs_run_name}_ft_") and "eval" in name]
            
            if ft == 'ft10':
                ft_run_name = [name for name in ft_run_name if name.endswith("eval_9998")]
            else:
                ft_run_name = [name for name in ft_run_name if not name.endswith("eval_9998")]
            
            if len(ft_run_name) == 0:
                print("!" * 50)
                print(f"Warning: Run {zs_run_name} not found in DataFrame. Using Random agent value.")
                print("!" * 50)
                
            all_vals = [df.loc[_run, metric] for _run in ft_run_name]
            row[metric] = np.max(all_vals)
            
        modified_data.append(row)
    # Convert the list of rows into a DataFrame
    # modified_df = pd.DataFrame(modified_data)
    # modified_df = modified_df.set_index("Method")
    modified_data = modified_data + other_rows
    
    return modified_data

def modified_df(df, run_dict, supp_df=None, supp_dict=None, include_ft=True):
    
    
    
    method_names = run_dict["mt_names"]
    zs_names = run_dict["zs_names"]
    zs_metrics = run_dict["zs_metrics"]
    
    # create a new data frame that has the same columns as the original
    # but with the method names as the first column
    modified_data = []
    
    all_columns = df.columns.tolist()

    # Iterate over each method name and populate the metrics
    for idx, (method_name, zs_name) in enumerate(zip(method_names, zs_names)):
        row = {"Method": method_name}
        
        # Fill zs_metrics values
        for metric in all_columns:
            result_type = "zs" if metric in zs_metrics else "mt"
            original_row_name = run_dict[result_type][idx]
            # check if both the row, and the metric exist in the DataFrame
            if original_row_name is None:
                # use supp df
                assert supp_df is not None, "supp_df must be provided if original_row_name is None"
                
                metric_val = supp_df.loc[supp_dict[result_type][idx], metric]
            # check if the original row name is in the DataFrame
            elif original_row_name not in df.index:
                
                print("!" * 50)
                print(f"Warning: Run {original_row_name} not found in DataFrame. Using Random agent value.")
                print("!" * 50)
                
                original_row_name = run_dict[result_type][-1]  # the random method value
                metric_val = df.loc[original_row_name, metric]
            
            else:
                metric_val = df.loc[original_row_name, metric]
                
                if pd.isna(metric_val):
                    # use the supp df
                    if metric == 'R14':
                        continue
                    
                    assert supp_df is not None, "supp_df must be provided if metric_val is NaN"
                    metric_val = supp_df.loc[supp_dict[result_type][idx], metric]            
            # pick the value from the original row, and metric column
            # row[metric] = df.loc[original_row_name, metric]
            row[metric] = metric_val
        
        modified_data.append(row)    
        
    if include_ft:
        # add the ft values to the modified data
        modified_data = get_ft_df(df, run_dict, modified_data, ft="ft10")
        # modified_data = get_ft_df(df, run_dict, modified_data, ft="ft30")
        # modified_data = modified_data[:-3] + modified_data[-2:] + modified_data[-3:-2]
        
    # Convert the list of rows into a DataFrame
    modified_df = pd.DataFrame(modified_data)
    modified_df = modified_df.set_index("Method")

    return modified_df      
    
class DataPointVisualizer:
    """
    A class for visualizing the relationship of a target data point
    to a set of original high-dimensional data points.
    """
    def __init__(self, original_data: np.ndarray, target_data: np.ndarray, original_names: list):
        """
        Initializes the visualizer with the original and target data.

        Args:
            original_data (np.ndarray): Array of shape (10, n_features).
            target_data (np.ndarray): Array of shape (1, n_features).
        """
        # if original_data.shape[0] != 10:
        #     raise ValueError("Original data should have 10 data points.")
        # if target_data.shape[0] != 1:
        #     raise ValueError("Target data should have 1 data point.")
        if original_data.shape[1] != target_data.shape[1]:
            raise ValueError("Original and target data must have the same number of features.")

        self.original_data = original_data
        self.original_names = original_names
        self.target_data = target_data
        self.all_data = np.vstack((original_data, target_data))
        self.n_original = original_data.shape[0]
        self.n_target = target_data.shape[0]
        self.n_features = original_data.shape[1]
        self._scaled_data = None
        self._scaled_original_data = None
        self._scaled_target_data = None
        self._scale_data()

    def _scale_data(self):
        """Scales the combined data using StandardScaler."""
        scaler = StandardScaler()
        self._scaled_data = scaler.fit_transform(self.all_data)
        self._scaled_original_data = self._scaled_data[:-1]
        self._scaled_target_data = self._scaled_data[-1]

    def plot_pca_elbow(self):
        """Generates a PCA elbow plot to determine the optimal number of components."""
        pca = PCA()
        pca.fit(self._scaled_data)
        explained_variance = pca.explained_variance_ratio_

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(explained_variance) + 1)),
                                 y=explained_variance,
                                 mode='lines+markers',
                                 name='Explained Variance'))
        fig.update_layout(title='PCA Elbow Plot',
                          xaxis_title='Number of Components',
                          yaxis_title='Explained Variance Ratio')
        fig.show()
        return explained_variance

    def plot_pca(self, save_dir=None):
        """Visualizes the data using PCA projection to 2 dimensions."""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self._scaled_data)
        pca_df = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'Type': ['Original'] * self.n_original + ['Target'] * self.n_target
        })
        colors = {'Original': 'blue', 'Target': 'red'}
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Type',
                         color_discrete_map=colors,
                        size=[10] * self.n_original + [20] * self.n_target,
                         title='PCA Projection')
        if save_dir:
            save_path = save_dir / "pca.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()

    def plot_tsne(self, n_iter: int = 300, random_state: int = 42, perplexity: int = 5, save_dir=None):
        """
        Visualizes the data using t-SNE projection to 2 dimensions.

        Args:
            n_iter (int): Number of iterations for optimization.
            random_state (int): Random seed for reproducibility.
            perplexity (int): Perplexity parameter for t-SNE (must be < n_samples).
            save_dir (Path): Directory to save the figure if provided.
        """
        perplexity = min(perplexity, self.all_data.shape[0] - 1)
        
        if perplexity >= self.all_data.shape[0]:
            raise ValueError(f"Perplexity must be less than the number of samples ({self.all_data.shape[0]}).")

        tsne = TSNE(n_components=2, random_state=random_state, n_iter=n_iter, perplexity=perplexity)
        tsne_result = tsne.fit_transform(self._scaled_data)
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_result[:, 0],
            'TSNE2': tsne_result[:, 1],
            'Type': ['Original'] * self.n_original + ['Target'] * self.n_target
        })
        colors = {'Original': 'blue', 'Target': 'red'}
        fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Type',
                         color_discrete_map=colors,
                         size=[10] * self.n_original + [20] * self.n_target,
                         title='t-SNE Projection')
        if save_dir:
            save_path = save_dir / "tsne.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()

    def plot_umap(self, n_neighbors: int = 5, random_state: int = 42, save_dir=None):
        """Visualizes the data using UMAP projection to 2 dimensions."""
        umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
        umap_result = umap.fit_transform(self._scaled_data)
        umap_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'Type': ['Original'] * self.n_original + ['Target'] * self.n_target
        })
        colors = {'Original': 'blue', 'Target': 'red'}
        fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', color='Type',
                         color_discrete_map=colors,
                         size=[10] * self.n_original + [20] * self.n_target,
                         title='UMAP Projection')
        if save_dir:
            save_path = save_dir / "umap.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()

    def plot_parallel_coordinates(self, save_dir=None):
        """Visualizes the data using a parallel coordinates plot."""
        feature_cols = [f'Feature_{i+1}' for i in range(self.n_features)]
        original_df = pd.DataFrame(self.original_data, columns=feature_cols)
        target_df = pd.DataFrame(self.target_data, columns=feature_cols)
        all_df = pd.concat([original_df, target_df], ignore_index=True)
        all_df['Type'] = ['Original'] * self.n_original + ['Target'] * self.n_target

        # Map 'Type' to numerical values for coloring
        color_map = {'Original': 0, 'Target': 1}
        all_df['Type_numeric'] = all_df['Type'].map(color_map)

        palette = sns.color_palette("Spectral")[::-1]  # Reverse the palette
        original_color = seaborn_color_to_rgb_string(palette[0])
        target_color = seaborn_color_to_rgb_string(palette[-1])

        # Use the numerical column for coloring
        fig = px.parallel_coordinates(
            all_df,
            dimensions=feature_cols,
            color='Type_numeric',
            # color_continuous_scale=['blue', 'red'],  # Map 0 to blue and 1 to red
            color_continuous_scale=[original_color, target_color],
            labels={'Type_numeric': 'Type'},
            title='Parallel Coordinates Plot'
        )
        
        if save_dir:
            save_path = save_dir / "parallel_coordinates.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()
        return fig
    
    def plot_parallel_coordinates_unique(self, save_dir=None):
        """Visualizes the data using a parallel coordinates plot, skipping features with constant values."""
        feature_cols = [f'Feature_{i+1}' for i in range(self.n_features)]
        original_df = pd.DataFrame(self.original_data, columns=feature_cols)
        target_df = pd.DataFrame(self.target_data, columns=feature_cols)
        all_df = pd.concat([original_df, target_df], ignore_index=True)
        all_df['Type'] = ['Original'] * len(self.original_data) + ['Target'] * len(self.target_data)

        # Remove features with constant values
        non_constant_cols = all_df.loc[:, all_df.nunique() > 1].columns.tolist()
        # Ensure 'Type' column is retained
        if 'Type' not in non_constant_cols:
            non_constant_cols.append('Type')
        filtered_df = all_df[non_constant_cols]

        # Map 'Type' to numerical values for coloring
        color_map = {'Original': 0, 'Target': 1}
        filtered_df['Type_numeric'] = filtered_df['Type'].map(color_map)

        # Generate colors for the plot
        palette = sns.color_palette("Spectral")[::-1]  # Reverse the palette
        original_color = seaborn_color_to_rgb_string(palette[0])
        target_color = seaborn_color_to_rgb_string(palette[-1])

        # Use the numerical column for coloring
        fig = px.parallel_coordinates(
            filtered_df,
            dimensions=[col for col in filtered_df.columns if col not in ['Type', 'Type_numeric']],
            color='Type_numeric',
            color_continuous_scale=[original_color, target_color],
            labels={'Type_numeric': 'Type'},
            title='Parallel Coordinates Plot (Filtered Features)'
        )
        
        if save_dir:
            save_path = save_dir / "parallel_coordinates_unique.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()
        
    def plot_parallel_coordinates_with_pca(self, save_dir=None):
        """Visualizes the data using a parallel coordinates plot with PCA-reduced features."""
        # Apply PCA to reduce the dimensionality of the data
        n_components = min(self.all_data.shape[0], self.all_data.shape[1])  # min(num_data, num_features)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self._scaled_data)

        # Create a DataFrame for the reduced data
        pca_feature_cols = [f'PCA_F{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_feature_cols)
        pca_df['Type'] = ['Original'] * self.n_original + ['Target'] * self.n_target

        # Map 'Type' to numerical values for coloring
        color_map = {'Original': 0, 'Target': 1}
        pca_df['Type_numeric'] = pca_df['Type'].map(color_map)

        # Generate colors for the plot
        palette = sns.color_palette("Spectral")[::-1]  # Reverse the palette
        original_color = seaborn_color_to_rgb_string(palette[0])
        target_color = seaborn_color_to_rgb_string(palette[-1])

        # Create the parallel coordinates plot
        fig = px.parallel_coordinates(
            pca_df,
            dimensions=pca_feature_cols,
            color='Type_numeric',
            color_continuous_scale=[original_color, target_color],
            labels={'Type_numeric': 'Type'},
            title='Parallel Coordinates Plot (PCA-Reduced Features)'
        )
        if save_dir:
            save_path = save_dir / "parallel_coordinates_pca.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()
        
    def plot_distance_heatmap(self, metric: str = 'euclidean', save_dir=None):
        """
        Visualizes the distances between the target point and the original points
        as a heatmap.

        Args:
            metric (str): The distance metric to use (e.g., 'euclidean', 'cosine').
                          Refer to scipy.spatial.distance.pdist for options.
        """
        if metric == 'euclidean':
            distances = np.linalg.norm(self.original_data - self.target_data, axis=1)
        else:
            distances = pdist(np.vstack((self.target_data, self.original_data)), metric=metric)[0:self.n_original]
        fig = go.Figure(data=go.Heatmap(z=[distances],
                            x=self.original_names,
                            colorscale='Viridis',
                            colorbar_title='Distance',
                            zmin=0 if metric == 'cosine' else 5,
                            zmax=1 if metric == 'cosine' else 20))
        fig.update_layout(title=f'Heatmap of Distances to Target Point ({metric.capitalize()})',
                          # turn off y axis labels
                            # yaxis=dict(title='Target Point', tickvals=[0], ticktext=['Target']),
                          )

        if save_dir:
            save_path = save_dir / f"distance_heatmap_{metric}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path), width=640, height=480)
        else:
            fig.show()          
        
        
def stack_heatmaps(original_data_list, target_data_list, original_names_list, metric='euclidean', subplot_names=None, save_dir=None):
    """
    Stacks heatmaps for multiple original-target pairs.

    Args:
        original_data_list (list): List of original data arrays (one per heatmap).
        target_data_list (list): List of target data arrays (one per heatmap).
        original_names_list (list): List of original names (one list per heatmap).
        metric (str): Distance metric to use (e.g., 'euclidean', 'cosine').
        save_dir (Path): Directory to save the stacked heatmap figure.
    """
    num_heatmaps = len(original_data_list)
    fig = make_subplots(
        rows=num_heatmaps,
        cols=1,
        # subplot_titles=[f"Heatmap {i+1}" for i in range(num_heatmaps)] if subplot_names is None else subplot_names,
        vertical_spacing=0.01,
        specs=[[{"type": "heatmap"}] for _ in range(num_heatmaps)],
    )

    annotations = []

    # three colors: cyan, light green, light red
    colors = sns.color_palette(palette='Paired')[:6][::2]
    colors = [seaborn_color_to_rgb_string(color, dark=0.8) for color in colors]
    selected_colors = [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2]]


    for i, (original_data, target_data, original_names) in enumerate(zip(original_data_list, target_data_list, original_names_list)):
        # Compute distances
        if metric == 'euclidean':
            distances = np.linalg.norm(original_data - target_data, axis=1)
        else:
            distances = 1-pdist(np.vstack((target_data, original_data)), metric=metric)[0:len(original_data)]

        print(f"Average {metric} distance for {subplot_names[i]}: {np.mean(distances):.2f}")

        # Create a heatmap with original names inside the cells
        z = [distances]
        text = [[name for name in original_names]]

        heatmap = go.Heatmap(
            z=z,
            text=text,
            # change the text size
            textfont=dict(size=16, family="Arial", color="white"),  # Set text color to white
            texttemplate="<b>%{text}</b><br>(%{z:.2f})",  # Show original name and distance
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text='Similarity',
                    font=dict(size=16)  # Increase colorbar title font size
                ),
                tickfont=dict(size=12),  # Increase colorbar tick font size
            ),
            # colorbar_title='Distance',
            zmin=0 if metric == 'cosine' else 5,
            zmax=1 if metric == 'cosine' else 20,
            showscale=(i == 0),  # Show colorbar only for the first heatmap
        )

        # Add the heatmap to the subplot
        fig.add_trace(heatmap, row=i + 1, col=1)

        # Hide axes for cleaner display
        fig.update_xaxes(visible=False, row=i + 1, col=1)
        fig.update_yaxes(visible=False, row=i + 1, col=1)
        
        if subplot_names:
            annotations.append(dict(
                text=subplot_names[i],
                xref="paper",
                yref="y" + str(i + 1),
                x=-0.01,  # Position the title slightly to the left of the heatmap
                y=0.05,  # Center the title vertically
                xanchor="right",
                yanchor="middle",
                textangle=-90,  # Rotate the text by 90 degrees
                showarrow=False,
                font=dict(size=16, color=selected_colors[i], family="Arial, bold")  # Set text color using selected_colors and make it bold
            ))

    # Update layout for the entire figure
    fig.update_layout(
        # title=f"Stacked Heatmaps ({metric.capitalize()} Distance)",
        title_x=0.5,
        # height=300 * num_heatmaps,  # Adjust height dynamically based on the number of heatmaps
        plot_bgcolor="white",
        annotations=annotations,
        # set margin to 0
        margin=dict(t=0, b=0, l=30, r=25),
        # modify the text size for annotations
    )

    # Save the figure if a save directory is provided
    if save_dir:
        save_path = save_dir / f"stacked_heatmaps_{metric}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path), width=900, height=350)
        print(f"Figure saved to {save_path}")

    # Show the figure
    # fig.show()
        
        
METHOD_GOOD_NAMES = {
    "Tf": "Transformer",
    "Mlp": "MLP",
    "Ind": "Single Embodiment",
    "Tf-ft10": "Transformer (ft-10k)",
    "Mlp-ft10": "MLP (ft-10k)",
    
    "Tf-ft30": "Transformer (ft-30k)",
    "Mlp-ft30": "MLP (ft-30k)",
}
        
        
class EvaluationVisualizer:
    """
    A class to visualize evaluation results from a pandas DataFrame.

    It generates bar graphs for average metric values per evaluation type
    and a scatter plot showing individual metric values with point size
    indicating the evaluation type.
    """
    def __init__(self, dataframe: pd.DataFrame, metric_set1: list, metric_set2: list, name="project"):
        """
        Initializes the EvaluationVisualizer.

        Args:
            dataframe (pd.DataFrame): DataFrame with methods as rows and metrics as columns.
            metric_set1 (list): List of column names belonging to the first evaluation type.
            metric_set2 (list): List of column names belonging to the second evaluation type.
        """
        self.df = dataframe
        self.metric_set1 = metric_set1
        self.metric_set2 = metric_set2
        
        # remove any columns that end in _reach (task_ur5 is the only one that has this)
        self.metric_set1 = [col for col in self.metric_set1 if not col.endswith("_reach")]
        self.metric_set2 = [col for col in self.metric_set2 if not col.endswith("_reach")]
        
        self.metric_set2_name = "Multi-Task Performance"
        self.metric_set2_shortname = "Multi-task"
        self.metric_set1_name = "Zero-Shot Performance"
        self.metric_set1_shortname = "Zero-shot"
        self.methods = self.df.index.tolist()
        self.num_methods = len(self.methods)
        self.name = name
        
    def construct_relative_df(self):
        # use the last method name as the reference
        # subtract the last method from all the other methods
        reference_method = self.methods[-1]
        reference_values = self.df.loc[reference_method]
        
        
        relative_df = self.df.copy()
        for method in self.methods:
            relative_df.loc[method] = relative_df.loc[method] - reference_values
            
        # remove the last method from the list
        relative_df = relative_df.drop(reference_method) 
        self.relative_df = relative_df
        self.relative_methods = relative_df.index.tolist()
        self.relative_num_methods = len(self.relative_methods)   
        
    def create_bar_graphs2(self, relative=False, ind="bar", group_by="eval_type"):
        if relative:
            df = self.relative_df
            methods = self.relative_methods
            num_methods = self.relative_num_methods
        else:
            df = self.df
            methods = self.methods
            num_methods = self.num_methods

        # Calculate averages for each metric set
                
        avg_set1 = df[self.metric_set1].mean(axis=1)
        avg_set2 = df[self.metric_set2].mean(axis=1)

        fig = go.Figure()

        if group_by == "eval_type":
            # x = [self.metric_set1_name, self.metric_set2_name]
            x = [self.metric_set2_shortname, self.metric_set1_shortname]
            
            # color: one for each method
            colors = sns.color_palette("GnBu", num_methods)  # Reverse the color palette
            colors = [seaborn_color_to_rgb_string(color) for color in colors]
            
            colors_test = sns.color_palette("OrRd", num_methods)  # Reverse the color palette
            colors_test = [seaborn_color_to_rgb_string(color) for color in colors_test]
            
            for idx, method in enumerate(methods[::-1]):            
                if ind != "bar" and method == "Ind":
                    continue
                else:
                    fig.add_trace(
                        go.Bar(name=f"{METHOD_GOOD_NAMES[method]}", 
                               x=x, y=[avg_set2[method], avg_set1[method]],
                            marker_color=[colors[idx], colors_test[idx]], text=[f"{avg_set2[method]:.2f}", f"{avg_set1[method]:.2f}"],)
                    )

            if ind != 'bar':
                # Add horizontal lines for each evaluation group
                fig.add_trace(go.Scatter(
                    x=[self.metric_set1_shortname, self.metric_set1_shortname],
                    y=[avg_set1["Ind"], avg_set1["Ind"]],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    name="Ind Avg (ZS)"
                ))
                fig.add_trace(go.Scatter(
                    x=[self.metric_set2_shortname, self.metric_set2_shortname],
                    y=[avg_set2["Ind"], avg_set2["Ind"]],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    name="Ind Avg (MT)"
                ))


        elif group_by == "method":
            x = methods
            # choose two colors: one darker of the other
            color1 = sns.color_palette("GnBu")[0]  # Reverse the color palette
            color2 = sns.color_palette("GnBu")[3]  # Reverse the color palette
            color1 = seaborn_color_to_rgb_string(color1)
            color2 = seaborn_color_to_rgb_string(color2)
            
            fig.add_trace(
                go.Bar(name=self.metric_set1_name, x=x, y=avg_set1, marker_color=color1, text=[f"{val:.3f}" for val in avg_set1])
            )
            fig.add_trace(
                go.Bar(name=self.metric_set2_name, x=x, y=avg_set2, marker_color=color2, text=[f"{val:.3f}" for val in avg_set2])
            )
            
        # Update layout
        fig.update_layout(
            barmode="group",
            yaxis=dict(
                zeroline=True,             # Enable the zero line
                zerolinewidth=2,        # Set the width of the zero line
                zerolinecolor='red'      # Set the color of the zero line
            )
        )

        # save the figure
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{self.name}_bar_plot2.png"), width=w, height=h)
        # fig.show()
        return fig
        
    
    def create_bar_graphs(self, relative=False):
        """
        Generates bar graphs showing the average of each metric set per method.
        """
        
        if relative:
            df = self.relative_df
            methods = self.relative_methods
            num_methods = self.relative_num_methods
            
        else:
            df = self.df
            methods = self.methods
            num_methods = self.num_methods
        
        
        avg_set1 = df[self.metric_set1].mean(axis=1)
        avg_set2 = df[self.metric_set2].mean(axis=1)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(self.metric_set1_name, self.metric_set2_name))
        
        colors = sns.color_palette("GnBu", num_methods)[::-1]  # Reverse the color palette
        colors = [seaborn_color_to_rgb_string(color) for color in colors]

        colors_test = sns.color_palette("GnBu", num_methods)[::-1]  # Reverse the color palette
        colors_test = [seaborn_color_to_rgb_string(color) for color in colors_test]

        fig.add_trace(go.Bar(x=methods, y=avg_set1, name=self.metric_set1_name,
                     marker_color=colors_test), row=1, col=1)
        fig.add_trace(go.Bar(x=methods, y=avg_set2, name=self.metric_set2_name,
                     marker_color=colors), row=1, col=2)

        fig.update_layout(title_text="Average Metric Values per Evaluation Type", showlegend=False)
        fig.update_xaxes(title_text="Agent")
        fig.update_yaxes(title_text="Average Value")
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{self.name}_bar_plot.png"), width=w, height=h)
                
        return fig
    
    def create_bar_graphs3(self, relative=True):
        if relative:
            df = self.relative_df
            methods = self.relative_methods
            num_methods = self.relative_num_methods
        else:
            df = self.df
            methods = self.methods
            num_methods = self.num_methods

        fig = go.Figure()

        # Generate colors for each method
        colors = sns.color_palette("GnBu", num_methods)  # Reverse the color palette
        colors = [seaborn_color_to_rgb_string(color) for color in colors][::-1]

        colors_test = sns.color_palette("OrRd", num_methods)  # Reverse the color palette
        colors_test = [seaborn_color_to_rgb_string(color) for color in colors_test][::-1]

        # create two subplots, one for each evaluation set
        fig = make_subplots(rows=1, cols=2, subplot_titles=(self.metric_set1_name, self.metric_set2_name),
                            column_widths=[1, 6])
        
        # for metric set 1
        for i, method in enumerate(methods):
            x_set1 = self.metric_set1
            y_set1 = df.loc[method, self.metric_set1].tolist()
            fig.add_trace(go.Bar(
                x=x_set1,
                y=y_set1,
                name=f"{method} (ZS)",
                marker_color=colors_test[i]
            ), row=1, col=1)
            
        # for metric set 2
        for i, method in enumerate(methods):
            x_set2 = self.metric_set2
            y_set2 = df.loc[method, self.metric_set2].tolist()
            fig.add_trace(go.Bar(
                x=x_set2,
                y=y_set2,
                name=f"{method} (MT)",
                marker_color=colors[i]
            ), row=1, col=2)

        fig.update_layout(
            barmode='group',  # Group bars for each method
            plot_bgcolor='ghostwhite',  # Sets the background color for the plot
            title=self.name,
            title_x=0.5,  # Center the title
            xaxis_title="Metrics",
            yaxis_title="Reward"
        )
        
        # update legend style (black border, white background)
        fig.update_layout(
            legend=dict(
                x=0.07,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="left",  # Anchor the legend's x position to the left
                yanchor="top",  # Anchor the legend's y position to the top
                font=dict(size=12),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
                bordercolor="black",  # Add a border color
                borderwidth=1,  # Set the border width
            )
        )
        
        # fig.show()
        
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{self.name}_bargraph3.png"), width=w, height=h)
        
        fig.show()
        return fig

    def create_scatter_plot(self, relative=False):
        """
        Generates a scatter plot showing individual metric values for each method.
        Point size is larger for metrics in the first evaluation set.
        """
        if relative:
            df = self.relative_df
            methods = self.relative_methods
            num_methods = self.relative_num_methods
        else:
            df = self.df
            methods = self.methods
            num_methods = self.num_methods
        
        fig = go.Figure()

        colors = sns.color_palette("GnBu", num_methods)[::-1]  # Reverse the color palette
        colors = [seaborn_color_to_rgb_string(color) for color in colors]

        for i, method in enumerate(methods):
            x_set1 = self.metric_set1
            y_set1 = df.loc[method, self.metric_set1].tolist()
            fig.add_trace(go.Scatter(x=x_set1, y=y_set1, mode='markers',
                                    marker=dict(size=15, color=colors[i]),
                                    name=f'{method} (ZS)'))

            x_set2 = self.metric_set2
            y_set2 = df.loc[method, self.metric_set2].tolist()
            fig.add_trace(go.Scatter(x=x_set2, y=y_set2, mode='markers',
                                    marker=dict(size=8, color=colors[i]),
                                    name=f'{method} (MT)',
                                    showlegend=False)) # Hide legend for the second set to avoid repetition

        fig.update_layout(title="Individual Metric Values per Method",
                          xaxis_title="Metrics",
                          yaxis_title="Value")
        fig.show()
            
    def create_scatter_plot2(self, relative=False):
        """
        Generates a scatter plot showing individual metric values for each method.
        Points are connected by lines for the same method, and point size is larger for metrics in the first evaluation set.
        """
        if relative:
            df = self.relative_df
            methods = self.relative_methods
            num_methods = self.relative_num_methods
        else:
            df = self.df
            methods = self.methods
            num_methods = self.num_methods

        fig = go.Figure()


        colors_test = sns.color_palette("OrRd", 2)  # Reverse the color palette
        colors_test = [seaborn_color_to_rgb_string(color) for color in colors_test]

        methods = ["Mlp", "Tf"]
        
        x_vals = [0, 10000, 30000]
        
        # select the index which is nan
        
        check1 = df.loc['Mlp-ft10', self.metric_set1[-1]]
        if np.isnan(check1):
            idx = 0
        else:
            idx = -1
        
        mlp_y_vals = [df.loc["Mlp", self.metric_set1[idx]], 
                  df.loc["Mlp-ft10", self.metric_set1[idx]], 
                  df.loc["Mlp-ft30", self.metric_set1[idx]], ]

        tf_y_vals = [df.loc["Tf", self.metric_set1[idx]],
                df.loc["Tf-ft10", self.metric_set1[idx]], 
                df.loc["Tf-ft30", self.metric_set1[idx]], ]
        
        reference = df.loc["Ind", self.metric_set1[idx]]
        
        # create a dashed line for the reference
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[reference] * len(x_vals),
            mode='lines',
            line=dict(color="black", width=1, dash="dash"),
            name="SE (test)"
        ))
    
        # create scatter points for MLP
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=mlp_y_vals,
            mode='markers+lines',
            marker=dict(size=15, color=colors_test[0]),
            line=dict(color=colors_test[0], width=2),
            name="MLP"
        ))
        
        # create scatter points for TF
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=tf_y_vals,
            mode='markers+lines',
            marker=dict(size=15, color=colors_test[1]),
            line=dict(color=colors_test[1], width=2),
            name="Transformer"
        ))
    
        fig.update_layout(
            title=self.name,
            # set location title to top center
            title_x=0.5,
            xaxis_title="# Finetuning Steps",
            yaxis_title="Task Score"
        )
        # fig.show()
        
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{self.name}_scatter_plot.png"), width=w, height=h)
        
        return fig
        
class SubplotVisualizer:
    """
    A class to visualize multiple scatter plots side by side for different pandas DataFrames.
    """
    def __init__(self, dataframes: list, metric_sets: list, names: list):
        """
        Initializes the SubplotVisualizer.

        Args:
            dataframes (list): List of pandas DataFrames, each representing a dataset.
            metric_sets (list): List of tuples, where each tuple contains two lists:
                                (metric_set1, metric_set2) for each DataFrame.
            names (list): List of names for each DataFrame, used as subplot titles.
        """
        self.dataframes = dataframes
        self.metric_sets = metric_sets
        self.names = names

        if len(dataframes) != len(metric_sets) or len(dataframes) != len(names):
            raise ValueError("The lengths of dataframes, metric_sets, and names must match.")
    
    def create_subplots(self, task='reach', *args, **kwargs):
        """
        Creates a horizontally stacked subplot visualization for the scatter plots.
        Ensures that only one legend (from the first plot) is shown.
        """
        # Create a subplot layout with one row and as many columns as there are DataFrames
        num_subplots = len(self.dataframes)
        n_cols = math.ceil(num_subplots / 2)


        self.dataframes = self.dataframes[3:]
        self.metric_sets = self.metric_sets[3:]
        self.names = self.names[3:]
        
        self.dataframes.pop(-2)
        self.metric_sets.pop(-2)
        self.names.pop(-2)

        fig = make_subplots(
            rows=1,
            cols=3,
            # subplot_titles=self.names[3:],
            subplot_titles=["Panda (I)", "EE-Arm (C)", "Arms (E)"],
            horizontal_spacing=0.05,  # Adjust spacing between subplots
            vertical_spacing=0.15,  # Adjust spacing between rows    
        )
        
        # Generate scatter plots for each DataFrame
        for i, (df, metric_set, name) in enumerate(zip(self.dataframes, self.metric_sets, self.names)):
            
            # if i<3:
            #     continue

            print(f"Plotting {name}...")

            idx = i
            
            metric_set1, metric_set2 = metric_set

            # Create an EvaluationVisualizer for the current DataFrame
            visualizer = EvaluationVisualizer(df, metric_set1, metric_set2, name=name)
            visualizer.construct_relative_df()

            # Generate the scatter plot figure
            project_fig = visualizer.create_scatter_plot2(*args, **kwargs)

            # Add traces to the subplot
            for trace in project_fig.data:
                # Hide the legend for all subplots except the first one
                # if "Avg" in trace.name:
                #     trace.showlegend = False    
                    
                # else:                
                trace.showlegend = (idx == 0)
                    
                # fig.add_trace(trace, row=i, col=i + 1)
                fig.add_trace(trace, row=(idx // n_cols) + 1, col=(idx % n_cols) + 1)

        # Update layout for the entire figure
        fig.update_layout(
            # title="Comparison of Scatter Plots Across DataFrames",
            title_x=0.5,  # Center the title
            # make y axis log scale
            # plot_bgcolor="ghostwhite",
            legend=dict(
                x=1.02,  # Position the legend to the right of the plot
                y=0.8,  # Align the legend to the top
                xanchor="left",  # Anchor the legend's x position to the left
                yanchor="top",  # Anchor the legend's y position to the top
                font=dict(size=8),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
                bordercolor="black",  # Add a border color
                borderwidth=1,  # Set the border width
            ),
            
            # font size for all text
            font=dict(size=12),
            margin=dict(l=1, r=1, t=20, b=0),  # Remove margins

        )

        # Show the combined figure
        fig.show()
            
        # Save the figure if a save directory is provided
        w, h = 630, 180
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"ft_scatter_{task}.pdf"), width=w, height=h)
        return fig
            
    
    def create_grouped_bargraphs(self, dfs, metric_sets, names, relative, metric="mt", category="cat"):
        visualizers = []
        new_metric_sets = []
        
        for df, metric_set, name in zip(dfs, metric_sets, names):
            # Create an EvaluationVisualizer for the current DataFrame
            visualizer = EvaluationVisualizer(df, metric_set[0], metric_set[1], name=name)
            new_metric_sets.append([visualizer.metric_set1, visualizer.metric_set2])
            visualizer.construct_relative_df()
            visualizers.append(visualizer)
        
        metric_sets = new_metric_sets
        
        if relative:
            methods = visualizer.relative_methods
            dfs = [visualizer.relative_df for visualizer in visualizers]
        else:
            methods = visualizer.methods
            dfs = [visualizer.df for visualizer in visualizers]

        if metric == "mt":
            if "Tf-ft10" in methods:
                methods.remove("Tf-ft10")
            if "Mlp-ft10" in methods:
                methods.remove("Mlp-ft10")
            # if "Tf"
            if "Tf-ft30" in methods:
                methods.remove("Tf-ft30")
            if "Mlp-ft30" in methods:
                methods.remove("Mlp-ft30")
            

        if metric == "mt":
            average_results = [df[metric_set[1]].mean(axis=1) for df, metric_set in zip(dfs, metric_sets)]
        elif metric == "zs":
            average_results = [df[metric_set[0]].mean(axis=1) for df, metric_set in zip(dfs, metric_sets)]


        if metric == "mt":        
            colors = sns.color_palette("GnBu", len(methods))  # Reverse the color palette
        else:
            colors = sns.color_palette("Oranges", len(methods)) # Reverse the color palette
        colors = [seaborn_color_to_rgb_string(color) for color in colors]

        group_colors = sns.color_palette("Paired")
        if category.lower() == "interpolation":
            group_color = seaborn_color_to_rgb_string(group_colors[0])
        elif category.lower() == "composition":
            group_color = seaborn_color_to_rgb_string(group_colors[1])
        else:
            group_color = seaborn_color_to_rgb_string(group_colors[2])

        fig = go.Figure()
    
        for idx, method in enumerate(methods[::-1]):
            fig.add_trace(
                go.Bar(
                    name=METHOD_GOOD_NAMES[method],
                    x=names,
                    y=[metric_result[method] for metric_result in average_results], 
                    marker_color=colors[idx],
                    text=[f"{metric_result[method]:.2f}" for metric_result in average_results],  # Add text labels
                    textfont=dict(size=12),  # Set text color to the group color
                )
            )        
            
        fig.update_traces(textfont_size=14) #, cliponaxis=False)
        
            
        # Update layout
        fig.update_layout(
            barmode="group",
            # bargap=0,
            # bargroupgap=0,
            title_text="Average Metric Values per Evaluation Type",
            showlegend=True,
            xaxis_title="Agent",
            yaxis_title="Average Value",
            xaxis=dict(
                tickfont=dict(color=group_color)
            ),
            # plot_bgcolor="ghostwhite",
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=0.4,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="top",  # Anchor the legend's y position to the top
                font=dict(size=12),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
                bordercolor="black",  # Add a border color
                borderwidth=1,  # Set the border width
            ),
            # margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
        )
        # Show the combined figure
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{category}_bar_plot.pdf"), width=w, height=h)
        # fig.show()
        return fig

    # def create_scatter_plots3(self, dfs, metric_sets, names, relative, category="cat"):
        
        
    #     fig = go.Figure()
        
        


    
    def create_subplots_bar2(self, legend=True, metric="mt", relative=False, group_indices=[0, 4, 6], task="reach", big=True, remove_group=None):
        category_names = ["Interpolation", "Composition", "Extrapolation"]
        
        if remove_group is not None:
            self.dataframes = self.dataframes[4:]
            self.metric_sets = self.metric_sets[4:]
            self.names = self.names[4:]
            group_indices.pop(remove_group)
            group_indices = [g - group_indices[0] for g in group_indices]

            category_names.pop(remove_group)

        num_subplots = len(self.dataframes)
        # there will be 3 subplots (according to the group_indices)
        
        n_cols = len(group_indices)
        n_rows = 1
        
        end_idx = group_indices[1:] + [num_subplots]

        n_plots_per_category = [end_idx[i] - group_indices[i] for i in range(len(group_indices))]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            column_widths=[0.5 * n for n in n_plots_per_category],
            # subplot_titles=category_names,    
            horizontal_spacing=0.04,  # Adjust spacing between subplots
            vertical_spacing=0.15,  # Adjust spacing between rows
        )
        
        end_idx = group_indices[1:] + [num_subplots]
        
        subplot_border_colors = {
            (1, 1): 'red',
            (1, 2): 'blue',
            (1, 3): 'green',
        }
        
        # each figure in the subplot is a grouped bar plot
        for subpplot_idx, start_idx in enumerate(group_indices):
            dfs = self.dataframes[start_idx:end_idx[subpplot_idx]]
            metric_sets = self.metric_sets[start_idx:end_idx[subpplot_idx]]
            names = self.names[start_idx:end_idx[subpplot_idx]]
            
            grouped_fig = self.create_grouped_bargraphs(dfs, metric_sets, names, relative=relative, metric=metric, category=category_names[subpplot_idx])
            # Add traces to the subplot
            for trace in grouped_fig.data:
                # Hide the legend for all subplots except the first one
                trace.showlegend = (subpplot_idx == 0) and legend
                fig.add_trace(trace, row=1, col=subpplot_idx + 1)
                # fig.add_trace(trace, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
                
                
        # add a single y-xis label for all subplots
        # Get the domain coordinates of the subplots
        # domain1 = fig.layout.grid.domain[0]  # For row=1, col=1
        # fig.update_layout(
        #     shapes=[
        #         go.layout.Shape(
        #             type="rect",
        #             xref="paper", yref="paper",
        #             x0=domain1.x[0], y0=domain1.y[0],
        #             x1=domain1.x[1], y1=domain1.y[1],
        #             line=dict(color="black", width=2),
        #         ),
        #     ]
        # )
                
                
        # Update layout for the entire figure
        yaxis_dict = {
            "zeroline": True,             # Enable the zero line
            "zerolinewidth": 2,        # Set the width of the zero line
            "zerolinecolor": 'black'      # Set the color of the zero line
        }
        fig.update_layout(
            # title="Comparison of Bar Graphs Across DataFrames",
            title_x=0.5,  # Center the title
            # make y axis log scale
            # plot_bgcolor="ghostwhite",
            font=dict(size=10 if big else 8),
            legend=dict(
                x=0.05,  # Position the legend to the right of the plot
                y=0.17,  # Align the legend to the top
                xanchor="left",  # Anchor the legend's x position to the left
                yanchor="top",  # Anchor the legend's y position to the top
                font=dict(size=10 if big else 8),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
                # bordercolor="black",  # Add a border color
                borderwidth=0.0,  # Set the border width
                # horizontal=True,
                orientation="h",
            ),
            yaxis=yaxis_dict,
            yaxis2=yaxis_dict,
            # yaxis3=yaxis_dict,
            margin=dict(l=1, r=1, t=0 if task == "push" else 5, b=0),  # Remove margins
            # annotations=[
            #     go.layout.Annotation(
            #         text="Success rate" if task == "push" else "Rel. Reward",
            #         xref="paper",
            #         x=0.0,
            #         y=0.5,
            #         yref='paper',
            #         textangle=-90,
            #     )
            # ]
        )       
        
        if n_cols == 3:
            fig.update_layout(
                yaxis3=yaxis_dict
            )
    
        if task == "push":
            fig.update_yaxes(range=[0.0, 1.0])
            # self.add_category_legend(fig)

        self.draw_bounding_box(fig)

        # fig.show()
        
        # save figure
        
        w, h = 75 * len(self.dataframes) - 30 * int(task=="push") , 280
        if task == "push":
            # h = 285
            h = 300
        
        val = 20
        fac=0.8
        w -= val * 2
        h -= val * 6
        w = int(w * fac)
        h = int(h * fac)
        
        if big:
            w = 800
            h = 200
        
        
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"bar_plot_{metric}_{task}{'_10' if big else ''}.pdf"), width=w, height=h)
        # fig.show()
        return fig
    
    
    def add_category_legend(self, fig):
        # Add custom legend entries (artificial legend)
        colors = sns.color_palette("Paired")[:6][::2]
        colors = [seaborn_color_to_rgb_string(color) for color in colors]
        
        custom_colors = colors
        custom_labels = ["Interpolation", "Composition", "Extrapolation"]

        for color, label in zip(custom_colors, custom_labels):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],  # Invisible data
                    mode='lines',
                    marker=dict(size=10, color=color),
                    name=label,
                    showlegend=True
                )
            )

        # Add whatever main plot content you want here
        # For example, a background image, or shapes, etc.
        fig.update_layout(
            margin=dict(l=1, r=1, t=0, b=0),  # Remove margins
            # title="Plot with Artificial Legend",
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="bottom",  # Anchor the legend's y position to the top
                font=dict(size=8),  # Set font size for the legend
                borderwidth=0.0,  # Set the border width
                bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
                # bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
            ),
        )
            
    def draw_bounding_box(self, fig):
                # Create a figure with subplots
        # Add some traces to the subplots
        # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
        # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[6, 5, 4]), row=1, col=2)
        # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[6, 5, 4]), row=1, col=3)
        
        # Define the bounding box properties for each subplot
        
        colors = sns.color_palette("Paired")[:6][::2]
        colors = [seaborn_color_to_rgb_string(color) for color in colors]
        layout = fig['layout']

        # Add rectangles (box boundaries) using shapes
        fig.update_layout(
            shapes=[
                # Red box around subplot (1,1)
                dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=layout['xaxis']['domain'][0], x1=layout['xaxis']['domain'][1],
                    y0=layout['yaxis']['domain'][0], y1=layout['yaxis']['domain'][1],
                    line=dict(color=colors[0], width=2)
                ),
                # Green box around subplot (1,2)
                dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=layout['xaxis2']['domain'][0], x1=layout['xaxis2']['domain'][1],
                    y0=layout['yaxis2']['domain'][0], y1=layout['yaxis2']['domain'][1],
                    line=dict(color=colors[1], width=2)
                ),
                # Blue box around subplot (2,1)
                dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=layout['xaxis3']['domain'][0], x1=layout['xaxis3']['domain'][1],
                    y0=layout['yaxis3']['domain'][0], y1=layout['yaxis3']['domain'][1],
                    line=dict(color=colors[2], width=2)
                ),
            ]
        )
            
        
    def create_subplots_bar(self, legend=True, *args, **kwargs):
        """
        Creates a horizontally stacked subplot visualization for the bar graphs.
        Ensures that only one legend (from the first plot) is shown.
        """
        # Create a subplot layout with one row and as many columns as there are DataFrames
        num_subplots = len(self.dataframes)
        # n_cols = math.ceil(num_subplots / 2)
        n_cols = num_subplots
        n_rows = 1
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=self.names,
            horizontal_spacing=0.02,  # Adjust spacing between subplots
            vertical_spacing=0.15,  # Adjust spacing between rows    
        )

        # Generate scatter plots for each DataFrame
        for i, (df, metric_set, name) in enumerate(zip(self.dataframes, self.metric_sets, self.names)):
            metric_set1, metric_set2 = metric_set

            # Create an EvaluationVisualizer for the current DataFrame
            visualizer = EvaluationVisualizer(df, metric_set1, metric_set2, name=name)
            visualizer.construct_relative_df()

            # Generate the scatter plot figure
            project_fig = visualizer.create_bar_graphs2(*args, **kwargs)

            # Add traces to the subplot
            for trace in project_fig.data:
                # Hide the legend for all subplots except the first one
                trace.showlegend = (i == 0) and legend           
                fig.add_trace(trace, row=1, col=i + 1)
                # fig.add_trace(trace, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
        # Update layout for the entire figure

        yaxis_dict = {
            "zeroline": True,             # Enable the zero line
            "zerolinewidth": 2,        # Set the width of the zero line
            "zerolinecolor": 'black'      # Set the color of the zero line
        }

        fig.update_layout(
            title="Comparison of Bar Graphs Across DataFrames",
            title_x=0.5,  # Center the title
            # make y axis log scale
            plot_bgcolor="ghostwhite",
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=0.4,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="top",  # Anchor the legend's y position to the top
                font=dict(size=12),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
                bordercolor="black",  # Add a border color
                borderwidth=1,  # Set the border width
            ),
            yaxis=yaxis_dict,
            yaxis2=yaxis_dict,
            yaxis3=yaxis_dict,
            yaxis4=yaxis_dict,
            yaxis5=yaxis_dict,
            yaxis6=yaxis_dict,
            yaxis7=yaxis_dict,
        )
        if len(self.dataframes) == 8:
            fig.update_layout(
                yaxis8=yaxis_dict,
            )
            fig.update_yaxes(range=[0.0, 1.0])
        
        # Show the combined figure
        # fig.show()
        
        # save figure
        
        w, h = 1280, 360
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / "bar_plo2.pdf"), width=w, height=h)
        
        fig.show()
        
        return fig
        
        