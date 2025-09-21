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

LEGEND_FONT_SIZE = 8
TRACE_FONT_SIZE = 8

def get_colors(task, metric, num_colors=2):
    if task == 'reach':
        if metric == 'mt':
            # colors = sns.color_palette("Spectral", 6)
            pastel1 = sns.color_palette("Pastel1", 6)
            pastel2 = sns.color_palette("Pastel2", 6)
            colors = [pastel2[5], pastel1[4]]
        else:
            copper = sns.color_palette("copper", 6)
            spectral = sns.color_palette("rainbow", 6)
            colors = [spectral[-2], copper[-2]]
            # colors = sns.color_palette("pink", 6)
            # colors = [colors[3], colors[1]]
    
    elif task == 'push':
        if metric == 'mt':
            colors = sns.color_palette("Blues", 6)
            colors = [colors[0], colors[1]]
        else:
            colors = sns.color_palette("RdYlBu", 6)
            colors = [colors[-2], colors[-1]]
    elif task == 'reach-ft':
        # mlp, mlp-ft10, mlp-ft30, mlp-ft50, tf, tf-ft10, tf-ft30, tf-ft50
        return colors
        raise NotImplementedError("Task not recognized")

    return colors

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

    mus = df.loc[:, [(m, 'mean') for m in metrics]].copy()
    sigmas = df.loc[:, [(m, 'std') for m in metrics]].copy()

    if normalize:
        for i, m in enumerate(metrics):
            if 'reach' in m:
                randval = df.loc['random', (m, 'mean')]
                diffikval = df.loc['diff-ik', (m, 'mean')]
                mus.iloc[:, i] = (mus.iloc[:, i] - randval) / (diffikval - randval)
                sigmas.iloc[:, i] = sigmas.iloc[:, i] / (diffikval - randval)
            # else: leave as is

    mu_sum = mus.sum(axis=1)
    sigma_sum = np.sqrt((sigmas ** 2).sum(axis=1))

    n = len(metrics)
    sigma_sum = sigma_sum / n
    mu_sum = mu_sum / n

    result = pd.DataFrame({"mean": mu_sum, "std": sigma_sum}, index=df.index)
    return result

    
def summarize_df(df, run_dict, normalize, only_zs=False):
    metric_set1 = run_dict['zs_metrics']
    metric_set2 = run_dict['mt_metrics']
    
    df_metric1 = combine_metrics(df, metric_set1, normalize=normalize)
    df_metric2 = combine_metrics(df, metric_set2, normalize=normalize)

    # naive_concat = pd.concat([df_metric1, df_metric2], axis=1, keys=['ZS', 'MT'])
    # print(naive_concat)
    
    # choose the correct rows for each dataframe
    zs_name_map = {orig: short for orig, short in zip(run_dict['zs'], run_dict['zs_names'])}
    mt_name_map = {orig: short for orig, short in zip(run_dict['mt'], run_dict['mt_names'])}
    
    available_zs = [zs_name for zs_name in run_dict['zs'] if zs_name in df.index]
    available_mt = [mt_name for mt_name in run_dict['mt'] if mt_name in df.index]

    df_zs = df_metric1.loc[available_zs]
    df_mt = df_metric2.loc[available_mt]

    df_zs_renamed = df_zs.rename(index={k: zs_name_map[k] for k in available_zs})
    df_mt_renamed = df_mt.rename(index={k: mt_name_map[k] for k in available_mt})
     
    if only_zs:
        df_zs_renamed.columns = pd.MultiIndex.from_product([['ZS'], df_zs_renamed.columns])
        return df_zs_renamed
    
    # now they should have same set of row names
    modified_df = pd.concat([df_zs_renamed, df_mt_renamed], axis=1, keys=['ZS', 'MT'])

    return modified_df


METHOD_GOOD_NAMES = {
    "Tf": "Transformer",
    "Mlp": "MLP",
    "Se-Mlp": "SE-Mlp",
    "Se-Tf": "SE-Tf",
    "Tf-ft10": "Tf (ft-10k)",
    "Mlp-ft10": "MLP (ft-10k)",
    "Tf-ft30": "Tf (ft-30k)",
    "Mlp-ft30": "MLP (ft-30k)",
    "Tf-ft50": "Tf (ft-30k)",
    "Mlp-ft50": "MLP (ft-30k)",
    "Tf-cont": "Tf (continuous)",
    "Tf-noCE": "Tf (no Critic EMA)",
    "Tf-noSL": "Tf (no Symlog)",
    "Mlp+SlCe": "MLP (+ Symlog + Critic EMA)",
}
         
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
        
    def create_per_task_bargraphs(self, df, task_name, methods, metrics=['mt', 'zs']):
        
        fig = go.Figure()
        
        # colors for metric 1 (darker shade)
        colors = sns.color_palette('pink', 6)
        colors = [seaborn_color_to_rgb_string(color) for color in colors]
        colors = [colors[-1], colors[-2]]

        metric_good_names = {
            'mt': 'Multi-task',
            'zs': 'Zero-shot'
        }

        for idx, metric in enumerate(metrics):
            mus = [df.loc[method, (metric.upper(), 'mean')] for method in methods]
            stds = [df.loc[method, (metric.upper(), 'std')] for method in methods]
            
            stds = [s if s not in (0, None, np.nan, 0.0) else None for s in stds]


            y_lower = np.array([m - s if s is not None else 0 for m, s in zip(mus, stds)])
            yerr_minus_clipped = np.where(y_lower < 0, np.array(mus), np.array(stds))

            text = [f"{m:.2f} ± {s:.3f}" if s is not None else f"{m:.2f}" for m, s in zip(mus, stds)]

            # stds = [s * std_scale if s is not None else None for s in stds ]
            # text = [f"{m:.2f} ± {s:.3f}" for m, s in zip(mus, stds)]

            fig.add_trace(
                go.Bar(
                    name=metric_good_names[metric],
                    # x=[METHOD_GOOD_NAMES[method] for method in methods],
                    x=methods,
                    y=mus,
                    error_y=dict(type='data', 
                                 array=stds,
                                 arrayminus=yerr_minus_clipped,
                                 thickness=0.75,
                                 color='black',
                                #  width=6.0,
                                 visible=True),
                    marker_color=colors[idx],
                    text=text,  # Add text labels
                    textangle=-90,                 # rotate text
                    insidetextanchor="middle",    # align at bar center
                    textfont=dict(),  # Set text color to the group color
                    # textposition='inside'  # Position text labels inside the bars
                )
            )


        fig.update_traces(textfont_size=TRACE_FONT_SIZE) #, cliponaxis=False)
        fig.update_layout(
            barmode="group"
        )
        # set the y axis to be between 0 and 1
        fig.update_yaxes(range=[0.0, None])
        
        # show y axis ticks and lines light gray
        fig.update_yaxes(showline=True, linecolor='lightgray', gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=1)
        
        # create a black bounding box around the plot
        fig.update_layout(
            barcornerradius=1,
            plot_bgcolor="white",
            margin=dict(l=1, r=1, t=1, b=20),  # Remove margins
            xaxis=dict(
                automargin=True
            ),
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="bottom",  # Anchor the legend's y position to the top
                font=dict(size=LEGEND_FONT_SIZE),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
                # horizontal orientation
                orientation="h",
            ),
        )
        
        fig.update_xaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        yaxis_dict = {
            "zeroline": True,             # Enable the zero line
            "zerolinewidth": 2,        # Set the width of the zero line
            "zerolinecolor": 'black',      # Set the color of the zero line
            'range': [0.0, None],
        }
        fig.update_yaxes(yaxis_dict)
        self.draw_bounding_box_individual(fig)
        
        w, h = 280, 200
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{task_name}_agents.pdf"), width=w, height=h)

        # fig.show()


    def create_scatter_plots(self, dfs, benchmark_task_names, methods):
        # scatter lines
        # there is only one df in dfs
        df = dfs[0]
        # need to create two lines one for mlp and one for tf, with x-vals as
        x_vals = [0, 10000, 30000, 50000]
        
        y1_vals_names = ['Mlp', 'Mlp-ft10', 'Mlp-ft30', 'Mlp-ft50']
        y2_vals_names = ['Tf', 'Tf-ft10', 'Tf-ft30', 'Tf-ft50']
        
        colors = get_colors(task='reach', metric='zs')
        colors = [seaborn_color_to_rgb_string(color) for color in colors]
        
        y1_mus = [df.loc[name, ('ZS', 'mean')] for name in y1_vals_names]
        y1_stds = [df.loc[name, ('ZS', 'std')] for name in y1_vals_names]

        y2_mus = [df.loc[name, ('ZS', 'mean')] for name in y2_vals_names]
        y2_stds = [df.loc[name, ('ZS', 'std')] for name in y2_vals_names]
        
        y_mus = [y1_mus, y2_mus]
        y_stds = [y1_stds, y2_stds]
        
        
        fig = go.Figure()
        
        for idx, method in enumerate(['Mlp', 'Tf']):
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_mus[idx],
                    error_y=dict(type='data',
                                 array=y_stds[idx],
                                 thickness=0.75,
                                 color='black',
                                 visible=True),
                    mode='lines+markers',
                    name=method,
                    line=dict(color=colors[idx], width=2),
                    marker=dict(size=TRACE_FONT_SIZE),
                    text=[f"{m:.2f} ± {s:.3f}" if s is not None else f"{m:.2f}" for m, s in zip(y_mus[idx], y_stds[idx])],
                )
            )
            
        fig.update_traces(textfont_size=TRACE_FONT_SIZE, textposition='top center') #, cliponaxis=False)
        fig.update_layout(
            title_text=benchmark_task_names[0],
            xaxis_title="# Fine-tuning Steps",
            yaxis_title="Average ZS Score",
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="bottom",  # Anchor the legend's y position to the top
                font=dict(size=LEGEND_FONT_SIZE),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
                orientation="h",
            ),
        )
        fig.update_yaxes(showline=True, linecolor='lightgray', gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=1)
        fig.update_xaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        
        yaxis_dict = {
            "zeroline": True,             # Enable the zero line
            "zerolinewidth": 2,        # Set the width of the zero line
            "zerolinecolor": 'black',      # Set the color of the zero line
            'range': [0.0, None],
        }
        fig.update_yaxes(yaxis_dict)
        fig.update_xaxes(dict(tickmode='array', tickvals=x_vals, ticktext=[str(x) for x in x_vals]))
        
        return fig
    

    def create_grouped_bargraphs(self, dfs, benchmark_task_names, methods, metric="mt", category="cat", task='reach', show_y_ticks=True, std_scale=1.0):
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

        # colors = sns.color_palette("Paired", 4)
        # if metric == "mt":        
        #     # colors = [colors[1], colors[3]]
        #     colors = sns.color_palette("GnBu", len(methods))
        # else:
        #     # colors = [colors[0], colors[2]]
        #     colors = sns.color_palette("Oranges", len(methods))
        
        colors = get_colors(task, metric, num_colors=len(methods))
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

            stds = [s if s not in (0, None, np.nan, 0.0) else None for s in stds]

            y_lower = np.array([m - s if s is not None else 0 for m, s in zip(means, stds)])
            yerr_minus_clipped = np.where(y_lower < 0, np.array(means), np.array(stds))

            text = [f"{m:.2f} ± {s:.3f}" if s is not None else f"{m:.2f}" for m, s in zip(means, stds) ]

            stds = [s * std_scale if s is not None else None for s in stds ]

            fig.add_trace(
                go.Bar(
                    name=METHOD_GOOD_NAMES[method],
                    x=benchmark_task_names,
                    y=means,
                    # width=0.35,
                    error_y=dict(type='data', 
                                 array=stds, 
                                 thickness=0.6,
                                 color='black',
                                 arrayminus=yerr_minus_clipped,
                                #  width=6.0,
                                 visible=True),
                    marker_color=colors[idx],
                    text=text,  # Add text labels
                    # textfont=dict(size=8),  # Set text color to the group color
                    # textposition='outside'  # Position text labels outside the bars
                    # textangle=-90,                 # rotate text
                    textposition='auto',
                    insidetextanchor="middle",    # align at bar center
                )
            )

        fig.update_traces(textfont_size=TRACE_FONT_SIZE) #, cliponaxis=False)

        # Update layout
        fig.update_layout(
            barcornerradius=2,
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
            # bargap=0.05,           # gap between bars of different x
            # bargroupgap=0       # gap between bars in a group
            # plot_bgcolor="ghostwhite",
            # legend=dict(
            #     x=1.0,  # Position the legend to the right of the plot
            #     y=1.0,  # Align the legend to the top
            #     xanchor="right",  # Anchor the legend's x position to the left
            #     yanchor="bottom",  # Anchor the legend's y position to the top
            #     font=dict(size=10),  # Set font size for the legend
            #     bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
            #     bordercolor="black",  # Add a border color
            #     borderwidth=1,  # Set the border width
            # ),
            # margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
        )
        fig.update_yaxes(range=[0.0, None])
        fig.update_xaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        
        # if show_y_ticks:
        #     fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        # else:
        #     fig.update_yaxes(showticklabels=False)
        
        # Show the combined figure
        w, h = 640, 480
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig.write_image(str(get_figures_dir() / f"{category}_{metric}.pdf"), width=w, height=h)
        # fig.show()
        return fig

    # def create_scatter_plots3(self, dfs, metric_sets, names, relative, category="cat"):

    def create_subplots_bar(self, legend=True, metric="mt", methods=['Mlp', 'Tf'], relative=False, group_indices=[0, 4, 6], task="reach", big=True, remove_group=None,
                            scatter=False):
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

        # c_width = 0.5 if task == 'push' else 0.1
        c_width = 0.5
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            column_widths=[c_width * n for n in n_plots_per_category],
            # horizontal_spacing=0.04,  # Adjust spacing between subplots
            horizontal_spacing=0.05 if "push" in task else 0.06,  
            vertical_spacing=0.15,  # Adjust spacing between rows
        )
        
        end_idx = group_indices[1:] + [num_subplots]

        # methods = ['Mlp', 'Tf']
        
        std_scale = 2.0 if (task == 'reach' and (metric == 'mt')) else 1.0
        
        # each figure in the subplot is a grouped bar plot
        for subpplot_idx, start_idx in enumerate(group_indices):
            dfs = self.dataframes[start_idx:end_idx[subpplot_idx]]
            names = self.names[start_idx:end_idx[subpplot_idx]]
            
            if scatter:
                grouped_fig = self.create_scatter_plots(dfs, names, methods)
            else:
                grouped_fig = self.create_grouped_bargraphs(dfs, names, methods, metric=metric, 
                                                        task=task,
                                                        category=category_names[subpplot_idx],
                                                        show_y_ticks=(subpplot_idx == 0), std_scale=std_scale)
            
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
            'range': [0.0, None],
            # ticks every 0.2 if task is push, else auto
            "dtick": 0.2 if task == 'push' else None,
            # "automargin": True,
        }
        fig.update_layout(
            # title=f"{task.capitalize()} Task - {'Multi-task' if metric == 'mt' else 'Zero-shot'} Performance",
            plot_bgcolor='white',  # inside all subplot axes
            paper_bgcolor='white',  # overall figure background
            title_x=0.5,  # Center the title
            barcornerradius=2,
            # font=dict(size=10),
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="bottom",  # Anchor the legend's y position to the top
                font=dict(size=LEGEND_FONT_SIZE),  # Set font size for the legend
                bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
                borderwidth=0.0,  # Set the border width
                orientation="h",
                itemwidth=30,
            ),
            # xaxis=dict(automargin=True),
            yaxis=yaxis_dict,
            yaxis2=yaxis_dict,
            margin=dict(l=1, r=1, t=1, b=30),  # Remove margins
        )
            
        # Update all x- and y-axes at once
        fig.update_yaxes(gridcolor='lightgray', gridwidth=1, griddash='dot')
        fig.update_yaxes(range=[0, None])
        # fig.update_traces(textfont_size=TRACE_FONT_SIZE) #, cliponaxis=False)
        fig.update_xaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        


        if n_cols == 3:
            fig.update_layout(
                yaxis3=yaxis_dict
            )

        # if 'reach' not in task:
        # self.add_category_legend(fig)

        self.draw_bounding_box(fig)

        # pdf_width_in = 6.0 if 'reach' in task else 7.0
        # pdf_height_in = 2.0 if 'reach' in task else 3.0
        pdf_width_in = 5.0 if 'reach' in task else 6.0
        pdf_height_in = 2.5
        dpi = 75
        fig_width = int(pdf_width_in * dpi)
        fig_height = int(pdf_height_in * dpi)

        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig_path = str(get_figures_dir() / f"{metric}_{task}{'_10' if big else ''}.pdf")
        fig.write_image(fig_path, width=fig_width, height=fig_height)
        print("Saved figure to", fig_path)
        # fig.show()
        return fig
    
    def create_finetuning_scatter_plot(self, *args, **kwargs):
        fig = self.create_subplots_bar(*args, **kwargs, scatter=True)
        # add labels for subplots
        names = ['Panda', 'EE-Arm', 'Arms']

        xpositions = [0.47, 0.5, 0.55]
        # manually add subtitles
        for i in range(3):
            fig.add_annotation(
                x=xpositions[i],
                y=-0.2,
                xref='x domain' if i == 0 else ('x2 domain' if i == 1 else 'x3 domain'),
                yref='y domain',
                text=names[i],
                showarrow=False,
                font=dict(size=TRACE_FONT_SIZE),
                align='center'
            )
            
        fig.update_layout(
            margin=dict(l=1, r=1, t=1, b=30),  # Remove margins
        )
        
        # add x-ticks at 0, 10k, 30k, 50k
        fig.update_xaxes(dict(tickmode='array', tickvals=[0, 10000, 30000, 50000], ticktext=['0', '10k', '30k', '50k']))
        
        fig.update_xaxes(tickfont=dict(size=TRACE_FONT_SIZE))
        fig.update_yaxes(tickfont=dict(size=TRACE_FONT_SIZE))

        pdf_width_in = 6.0
        pdf_height_in = 3.0
        dpi = 60
        fig_width = int(pdf_width_in * dpi)
        fig_height = int(pdf_height_in * dpi)
        get_figures_dir().mkdir(parents=True, exist_ok=True)
        fig_path = str(get_figures_dir() / f"finetuning_scatter.pdf")
        fig.write_image(fig_path, width=fig_width, height=fig_height)
        print("Saved figure to", fig_path)
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
                    marker=dict(size=TRACE_FONT_SIZE, color=color),
                    name=label,
                    showlegend=True
                )
            )

        fig.update_layout(
            margin=dict(l=1, r=1, t=0, b=0),  # Remove margins
            # title="Plot with Artificial Legend",
            legend=dict(
                x=1.0,  # Position the legend to the right of the plot
                y=1.0,  # Align the legend to the top
                xanchor="right",  # Anchor the legend's x position to the left
                yanchor="bottom",  # Anchor the legend's y position to the top
                font=dict(size=LEGEND_FONT_SIZE),  # Set font size for the legend
                borderwidth=0.0,  # Set the border width
                bgcolor="rgba(255, 255, 255, 0.0)",  # Set a semi-transparent background for the legend
                # bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent background for the legend
            ),
        )
            
    def draw_bounding_box(self, fig):
        # colors = sns.color_palette("Paired")[:6][::2]
        colors = [
            sns.color_palette("GnBu")[-1],
            sns.color_palette("BuGn")[-1],
            sns.color_palette('Oranges_d')[-1]
        ]
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
        

    def draw_bounding_box_individual(self, fig):
        # Draw bounding boxes around figure
        # there are no subplots only a single figure
        # colors = sns.color_palette("Paired")[:6][::2]
        colors = [
            sns.color_palette("GnBu")[-1],
        ]
        colors = [seaborn_color_to_rgb_string(color) for color in colors]
        color = colors[0]
        
        # Draw a rectangle around the **full data area**
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x",  # data coordinates
                    yref="y",
                    x0=-0.5,
                    x1=5.5,
                    y0=0.0,  # start of y-axis
                    y1=0.35,
                    line=dict(color=color, width=2)
                ),
            ]
        )