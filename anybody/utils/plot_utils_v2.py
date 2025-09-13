import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.colors as mpl
import plotly.io as pio

# pio.kaleido.scope.mathjax = None

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

def combine_metrics(df, metrics, normalize=False):
    mus = df.loc[:, [(m, 'mean') for m in metrics]]
    sigmas = df.loc[:, [(m, 'std') for m in metrics]]
    
    # for reach tasks, we want to normalize the metrics to be between 0 and 1
    # use Rand index as 0
    # use diff-ik index as 1
    
    # get the values of other agents by scaling between rand and diff-ik
    if normalize:
        randvals = df.loc['Rand', [(m, 'mean') for m in metrics]]
        diffikvals = df.loc['diff-ik', [(m, 'mean') for m in metrics]]
        mus = (mus - randvals) / (diffikvals - randvals)
        sigmas = sigmas / (diffikvals - randvals).values

    mu_sum = mus.sum(axis=1)
    sigma_sum = np.sqrt((sigmas ** 2).sum(axis=1))
    
    n = len(metrics)
    sigma_sum = sigma_sum / n
    mu_sum = mu_sum / n
    
    result = pd.DataFrame({"mean": mu_sum, "std": sigma_sum}, index=df.index)
    return result


def modify_method_names(df, run_dict):
    df_method_names = df.index.tolist()
    
    original_names = run_dict['mt']
    short_names = run_dict['mt_names']
    
    rename_dict = {orig: short for orig, short in zip(original_names, short_names)}
    
    for df_mtd in df_method_names:
        if df_mtd not in rename_dict:
            rename_dict[df_mtd] = df_mtd # keep the same name if not found in the dict
            
    return df.rename(index=rename_dict)
    
def summarize_df(df, run_dict, normalize):
    metric_set1 = run_dict['zs_metrics']
    metric_set2 = run_dict['mt_metrics']
    
    df_metric1 = combine_metrics(df, metric_set1, normalize=normalize)
    df_metric2 = combine_metrics(df, metric_set2, normalize=normalize)
    
    modified_df = pd.concat([df_metric1, df_metric2], axis=1, keys=['ZS', 'MT'])
    
    return modified_df


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
                
        import pdb; pdb.set_trace()
                
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
    def __init__(self, dataframes: list, names: list):
        """
        Initializes the SubplotVisualizer.

        Args:
            dataframes (list): List of pandas DataFrames, each representing a dataset.
            metric_sets (list): List of tuples, where each tuple contains two lists:
                                (metric_set1, metric_set2) for each DataFrame.
            names (list): List of names for each DataFrame, used as subplot titles.
        """
        self.dataframes = dataframes
        self.names = names
            
    def create_grouped_bargraphs(self, dfs, benchmark_task_names, methods, metric="mt", category="cat"):
        # methods = dfs[0].index.tolist()
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
    
        for idx, method in enumerate(methods):
            means = [df.loc[method][metric.upper(), 'mean'] for df in dfs]
            stds = [df.loc[method][metric.upper(), 'std'] for df in dfs]
            
            fig.add_trace(
                go.Bar(
                    name=METHOD_GOOD_NAMES[method],
                    x=benchmark_task_names,
                    y=means,
                    error_y=dict(type='data', 
                                 array=stds, 
                                 thickness=0.75,
                                 color='black',
                                #  width=6.0,
                                 visible=True),
                    marker_color=colors[idx],
                    text=[f"{mu:.2f}" for mu in means],  # Add text labels
                    textfont=dict(size=12),  # Set text color to the group color
                    textposition='outside'  # Position text labels outside the bars
                )
            )        
            
        fig.update_traces(textfont_size=8) #, cliponaxis=False)
            
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
        # w, h = 640, 480
        # get_figures_dir().mkdir(parents=True, exist_ok=True)
        # fig.write_image(str(get_figures_dir() / f"{category}_bar_plot.pdf"), width=w, height=h)
        # fig.show()
        return fig

    # def create_scatter_plots3(self, dfs, metric_sets, names, relative, category="cat"):
    
    def create_subplots_bar2(self, legend=True, metric="mt", relative=False, group_indices=[0, 4, 6], task="reach", big=True, remove_group=None):
        category_names = ["Interpolation", "Composition", "Extrapolation"]
        
        if remove_group is not None:
            self.dataframes = self.dataframes[4:]
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
            horizontal_spacing=0.04,  # Adjust spacing between subplots
            vertical_spacing=0.15,  # Adjust spacing between rows
        )
        
        end_idx = group_indices[1:] + [num_subplots]
        
        subplot_border_colors = {
            (1, 1): 'red',
            (1, 2): 'blue',
            (1, 3): 'green',
        }
        
        methods = ['Mlp', 'Tf']
        
        # each figure in the subplot is a grouped bar plot
        for subpplot_idx, start_idx in enumerate(group_indices):
            dfs = self.dataframes[start_idx:end_idx[subpplot_idx]]
            names = self.names[start_idx:end_idx[subpplot_idx]]
            
            
            grouped_fig = self.create_grouped_bargraphs(dfs, names, methods, metric=metric, category=category_names[subpplot_idx])
            # Add traces to the subplot
            for trace in grouped_fig.data:
                # Hide the legend for all subplots except the first one
                trace.showlegend = (subpplot_idx == 0) and legend
                fig.add_trace(trace, row=1, col=subpplot_idx + 1)
                # fig.add_trace(trace, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
                
        # Update layout for the entire figure
        yaxis_dict = {
            "zeroline": True,             # Enable the zero line
            "zerolinewidth": 2,        # Set the width of the zero line
            "zerolinecolor": 'black',      # Set the color of the zero line
            # "automargin": True,
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
                borderwidth=0.0,  # Set the border width
                orientation="h",
            ),
            xaxis=dict(automargin=True),
            yaxis=yaxis_dict,
            yaxis2=yaxis_dict,
            margin=dict(l=1, r=1, t=1, b=30),  # Remove margins
        )

        if n_cols == 3:
            fig.update_layout(
                yaxis3=yaxis_dict
            )
    
        # if task == "push":
        fig.update_yaxes(range=[0.0, 1.0])
            # self.add_category_legend(fig)

        self.draw_bounding_box(fig)

        pdf_width_in = 5.0
        pdf_height_in = 2.0
        dpi = 75
        fig_width = int(pdf_width_in * dpi)
        fig_height = int(pdf_height_in * dpi)

        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"bar_plot_{metric}_{task}{'_10' if big else ''}.pdf"), width=fig_width, height=fig_height)
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