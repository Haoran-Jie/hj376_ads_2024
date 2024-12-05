import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew, kurtosis
from scipy.stats import probplot
import scipy.stats as stats



def plot_boxplot_price_by_property_type(final_matched_df):
    """Plot the boxplot of price for each property type."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_matched_df, x='property_type', y='price')
    plt.yscale('log')
    plt.title("Boxplot of Price by Property Type (Log Scale)")
    plt.xlabel("Property Type")
    plt.ylabel("Price (log scale)")
    plt.show()

def plot_geodata(geodf, edges, area, xlim, ylim, condition=None, label = None, title = "Map Visualisation"):
    """
    Plot GeoDataFrame with nodes, edges, and area.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    area.plot(ax=ax, facecolor="white")
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if condition is not None:
        geodf[condition].plot(ax=ax, color="red", alpha=0.7, markersize=10, label=label)
        geodf[~condition].plot(ax=ax, color="blue", alpha=0.7, markersize=10, label="Not " + label)
    else:
        geodf.plot(ax=ax, color="red", alpha=0.7, markersize=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.legend()

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_regularization_results(ridge_df, lasso_df, baseline_rmse, baseline_r2):
    """
    Plot the regularization results for Ridge and Lasso models.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # RMSE Plot
    axs[0].plot(ridge_df['alpha'], ridge_df['rmse'], marker='o', label='Ridge RMSE', color='b')
    axs[0].plot(lasso_df['alpha'], lasso_df['rmse'], marker='o', label='Lasso RMSE', color='r')
    axs[0].axhline(baseline_rmse, color='g', linestyle='--', label='Baseline RMSE')
    axs[0].set_xscale('log')
    axs[0].set_ylabel('RMSE')
    axs[0].set_xlabel('Alpha (Regularization Strength)')
    axs[0].legend()
    axs[0].grid(True)

    # R² Plot
    axs[1].plot(ridge_df['alpha'], ridge_df['r2'], marker='o', label='Ridge R²', color='b')
    axs[1].plot(lasso_df['alpha'], lasso_df['r2'], marker='o', label='Lasso R²', color='r')
    axs[1].axhline(baseline_r2, color='g', linestyle='--', label='Baseline R²')
    axs[1].set_xscale('log')
    axs[1].set_ylabel('R²')
    axs[1].set_xlabel('Alpha (Regularization Strength)')
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

def plot_coefficients(coefficients_df):
    """
    Plot the coefficients for each feature across age groups.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in coefficients_df.columns:
        ax.plot(range(100), coefficients_df[col], label=col)

    ax.set_xlabel("Age Group")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Model Coefficients Across Age Groups")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def calculate_and_visualise_correlations(
    data, analyses, grid_size=None, figsize=(16, 10)
):
    import math
    """
    Generalized function to calculate and visualize correlations in subplots.

    Parameters:
        data: pandas DataFrame containing the data.
        analyses: List of dictionaries, each specifying:
            - x_col: Column for the x-axis.
            - y_col: Column for the y-axis.
            - filtering_condition (optional): Filtering condition for the data.
            - x_log (optional): Whether to use a log scale for the x-axis.
            - y_log (optional): Whether to use a log scale for the y-axis.
            - hue_col (optional): Column for hue in scatterplots.
        grid_size (optional): Tuple specifying (rows, cols) for the subplot grid. Automatically calculated if not provided.
        figsize: Tuple specifying the overall figure size.
    """
    # Determine grid size automatically if not specified
    num_analyses = len(analyses)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_analyses))
        rows = math.ceil(num_analyses / cols)
    else:
        rows, cols = grid_size

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten in case of a grid

    for i, analysis in enumerate(analyses):
        if i >= len(axes):  # Avoid index out of range in case of excess analyses
            break

        # Extract parameters for the analysis
        x_col = analysis["x_col"]
        y_col = analysis["y_col"]
        filtering_condition = analysis.get("filtering_condition", None)
        x_log = analysis.get("x_log", False)
        y_log = analysis.get("y_log", False)
        hue_col = analysis.get("hue_col", None)

        # Apply filtering if specified
        filtered_data = data
        if filtering_condition is not None:
            filtered_data = filtered_data[filtering_condition]

        # Calculate correlation
        if not filtered_data.empty:
            corr = filtered_data[x_col].corr(filtered_data[y_col])
        else:
            corr = float("nan")
            print(f"No data left after filtering for analysis {i+1}.")

        # Plot scatter and regression
        sns.scatterplot(
            data=filtered_data, x=x_col, y=y_col, hue=hue_col, alpha=0.6, ax=axes[i]
        )
        sns.regplot(
            data=filtered_data,
            x=x_col,
            y=y_col,
            scatter=False,
            ax=axes[i],
            color="blue",
        )

        # Log scale adjustments
        if x_log:
            axes[i].set_xscale("log")
        if y_log:
            axes[i].set_yscale("log")

        # Titles and labels
        axes[i].set_title(f"{y_col} vs {x_col}\nCorrelation: {corr:.2f}")
        axes[i].set_xlabel(x_col + " (log scale)" if x_log else x_col)
        axes[i].set_ylabel(y_col + " (log scale)" if y_log else y_col)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def train_and_visualize_feature_importance(X, y, target_name, top_n=20):
    """
    Train a Random Forest Regressor and visualize feature importances.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        target_name (str): Name of the target variable for titles.
        top_n (int): Number of top features to visualize.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n), hue = 'Feature')
    plt.title(f'Top {top_n} Feature Importances for {target_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return feature_importances

def plot_histograms_and_boxplots(data_list, titles, xlabels, hist_color='skyblue', box_color='skyblue', figsize=(15, 5)):
    """
    Plot histograms and boxplots for multiple datasets with statistical annotations.
    
    Parameters:
        data_list (list of pd.Series): List of data arrays or pandas Series to plot.
        titles (list of str): Titles for each plot.
        xlabels (list of str): Labels for the x-axis for each plot.
        hist_color (str, optional): Color for histogram bars. Default is 'skyblue'.
        box_color (str, optional): Color for boxplot boxes. Default is 'skyblue'.
        figsize (tuple, optional): Size of the figure. Default is (15, 5).
    """
    if len(data_list) != len(titles) or len(data_list) != len(xlabels):
        raise ValueError("Length of data_list, titles, and xlabels must match.")

    # Prepare data for histograms
    fig, axs = plt.subplots(1, len(data_list), figsize=figsize)
    
    for i, data in enumerate(data_list):
        axs[i].hist(data, bins=50, color=hist_color, edgecolor='black')

        # calculate skewness and kurtosis and annotate in the top-right corner
        skewness = data.skew()
        kurtosis = data.kurtosis()
        stats_text = f"Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}"
        axs[i].text(
            0.95, 0.95, stats_text,
            transform=axs[i].transAxes,
            fontsize=9, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        axs[i].set_yscale('log')
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel('Frequency (log scale)')
        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()

    # Prepare data for boxplots
    fig, axs = plt.subplots(1, len(data_list), figsize=figsize)

    for i, data in enumerate(data_list):
        # Plot boxplot
        box = axs[i].boxplot(
            data, vert=False, patch_artist=True,
            boxprops=dict(facecolor=box_color, color='black'),
            flierprops=dict(marker='o', color='#6d6875', markersize=5)
        )
        axs[i].set_xscale("log")

        # Calculate statistics
        data_clean = data.dropna() if isinstance(data, pd.Series) else data[~np.isnan(data)]
        q1, median, q3 = np.percentile(data_clean, [25, 50, 75])
        min_val = np.min(data_clean)
        max_val = np.max(data_clean)
        
        # Annotate statistics in the top-right corner
        stats_text = (f"Min: {min_val:.2f}\n"
                      f"Q1: {q1:.2f}\n"
                      f"Median: {median:.2f}\n"
                      f"Q3: {q3:.2f}\n"
                      f"Max: {max_val:.2f}")
        axs[i].text(
            0.95, 0.95, stats_text, transform=axs[i].transAxes,
            fontsize=9, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        # Customize appearance
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(f"Boxplot of {xlabels[i]}")

    plt.tight_layout()
    plt.show()

def visualize_feature_correlations(correlations, title, color):
    """Visualize the correlation of features with a target variable."""
    plt.figure(figsize=(13, 7))
    correlations.drop(correlations.index[0]).plot(kind="bar", color=color)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Correlation")
    plt.show()


def plot_feature_target_pairplots(df, features, target, title, composite_features = None, ncols=3, log_y=False):
    """
    Create subplots with pairplots of each feature against the target variable.

    Parameters:
        df (pd.DataFrame): The DataFrame containing features and target variable.
        features (list): List of feature columns to plot.
        target (str): Target column name.
        title (str): Title of the plot grid.
        ncols (int): Number of columns in the grid of subplots.
    """
    if composite_features is None:
        composite_features = {}
    n_features = len(features) + len(composite_features)
    nrows = (n_features + ncols - 1) // ncols  # Calculate rows to fit all features

    df_tmp = df.copy()
    for composite_feature in composite_features.keys():
        df_tmp[composite_feature] = df_tmp[composite_features[composite_feature]].sum(axis=1)

    features += list(composite_features.keys())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for idx, feature in enumerate(features):
        df_tmp_1 = df_tmp[df_tmp[feature] > 0]
        # remove the top 1% of the feature values for better visualization
        if df_tmp_1[feature].dtype == 'float64':
            df_tmp_1 = df_tmp_1[df_tmp_1[feature] < df_tmp_1[feature].quantile(0.99)]
        sns.scatterplot(x=df_tmp_1[feature], y=df_tmp_1[target], ax=axes[idx], alpha=0.6)
        axes[idx].set_title(f"{feature} vs {target}")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target)
        if log_y:
            axes[idx].set_yscale('log')

    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout and title
    fig.suptitle(title, fontsize=16, y=1.003)
    fig.tight_layout()
    plt.show()


def plot_top_feature_correlations(correlations_dict, levels, title_suffix, ylabel="Correlation", top_n=5):
    """
    Plots the top feature correlations for multiple levels as subplots.

    Parameters:
    - correlations_dict (dict): A dictionary with level names as keys and correlation Series as values.
    - levels (list): List of levels to include in the plot, e.g., ['oa', 'lsoa', 'city'].
    - title_suffix (str): Suffix for subplot titles, e.g., "with L15 Proportion".
    - ylabel (str): Label for the y-axis. Default is "Correlation".
    - top_n (int): Number of top correlations to plot for each level. Default is 5.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt

    # Set up subplots
    fig, axs = plt.subplots(1, len(levels), figsize=(6 * len(levels), 6), sharey=True)

    for i, level in enumerate(levels):
        # Extract correlations for the level
        correlations = correlations_dict.get(level)
        if correlations is not None:
            # Plot the top N correlations
            correlations.head(top_n).plot(kind='bar', ax=axs[i], color='skyblue', edgecolor='black')
            axs[i].set_title(f"Top {top_n} Feature Correlations {title_suffix} ({level.upper()} Level)")
            axs[i].set_xlabel('Feature Group')
            axs[i].set_ylabel(ylabel if i == 0 else "")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')

            # Add grid
            axs[i].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_correlation_improvement(
    original_correlations, transformed_correlations, top_n=5, title="", ylabel="Correlation Improvement"
):
    """
    Plot comparison of features with the largest improvement in correlation after transformation.

    Parameters:
    - original_correlations (pd.Series): Correlations before transformation.
    - transformed_correlations (pd.Series): Correlations after transformation.
    - top_n (int): Number of features with the largest improvement to consider.
    - title (str): Title of the plot.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None (Displays the plot).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate the improvement in correlation
    correlation_improvement = transformed_correlations - original_correlations

    # Find the top N features with the largest improvement
    top_features = correlation_improvement.abs().nlargest(top_n).index

    # Extract relevant data
    comparison_df = pd.DataFrame({
        "Before Transformation": original_correlations.loc[top_features],
        "After Transformation": transformed_correlations.loc[top_features],
        "Improvement": correlation_improvement.loc[top_features]
    }).sort_values("Improvement", ascending=False)

    # Plot the comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width for grouping
    bar_width = 0.4
    x = np.arange(len(comparison_df))

    # Plot bars
    ax.bar(x - bar_width / 2, comparison_df["Before Transformation"], bar_width, label="Before Transformation", color="skyblue", edgecolor="black")
    ax.bar(x + bar_width / 2, comparison_df["After Transformation"], bar_width, label="After Transformation", color="orange", edgecolor="black")

    # Add labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
    ax.legend()

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_turnout_distribution(
    election_results,
    turnout_column="Turnout_rate",
    party_column="First_party",
    party_colors=None,
    bins=np.arange(40, 76, 1),
    title="Distribution of Turnout Rate by Winning Party",
    figsize=(12, 8)
):
    """
    Plot a stacked histogram with KDE and statistics for election turnout rates.
    
    Args:
        election_results (pd.DataFrame): DataFrame containing election data.
        turnout_column (str): Column name for turnout rates.
        party_column (str): Column name for party affiliation.
        party_colors (dict): Dictionary mapping party names to colors.
        bins (np.array): Array of bin edges for the histogram.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
    
    Returns:
        None
    """
    if party_colors is None:
        party_colors = {"Con": "#2475C0", "Lab": "#DA302F", "LD": "#F7A938", "Other": "#8E8D8D"}

    # Categorize winning party
    election_results["Winning_party"] = election_results[party_column].apply(
        lambda x: "Other" if x not in ["Con", "Lab", "LD"] else x
    )
    
    # Define party categories
    party_categories = ["Lab", "Con", "LD", "Other"]
    stacked_data = [
        election_results.loc[election_results["Winning_party"] == party, turnout_column]
        for party in party_categories
    ]

    # Calculate statistics
    mean = election_results[turnout_column].mean()
    std = election_results[turnout_column].std()
    skewness = skew(election_results[turnout_column])
    kurt = kurtosis(election_results[turnout_column])

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the stacked histogram
    counts, _, _ = ax1.hist(
        stacked_data,
        bins=bins,
        stacked=True,
        color=[party_colors[party] for party in party_categories],
        edgecolor="black",
        alpha=0.8,
        label=party_categories,
    )

    ax1.set_xlabel("Turnout Rate (%)", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.tick_params(axis="y")

    # Add KDE plot on secondary axis
    ax2 = ax1.twinx()
    sns.kdeplot(
        election_results[turnout_column],
        color="black",
        linewidth=2,
        label="KDE",
        ax=ax2
    )

    ax2.set_ylabel("Density", fontsize=14, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Add title
    plt.title(title, fontsize=16, fontweight="bold")

    # Annotate statistics
    stats_text = f"Mean: {mean:.2f}%\nStd Dev: {std:.2f}%\nSkewness: {skewness:.2f}\nKurtosis: {kurt:.2f}"
    ax1.annotate(stats_text, xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                 fontsize=12)

    # Add legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, title="Legend", fontsize=12, title_fontsize=14)

    # Add grid and layout adjustments
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_qq(data, title="QQ Plot", xlabel="Theoretical Quantiles", ylabel="Ordered Values", 
            dot_color="#2475C0", line_color="#DA302F"):
    """
    Creates a QQ plot for the given data with improved formatting and customization options.
    
    Parameters:
    - data (array-like): The data to be plotted.
    - title (str): The title of the plot.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - dot_color (str): Color of the dots in the QQ plot.
    - line_color (str): Color of the diagonal reference line in the QQ plot.

    Returns:
    - None: Displays the QQ plot.
    """
    # Create the QQ plot
    plt.figure(figsize=(10, 6))
    qq = probplot(data, dist="norm")
    theoretical_quantiles, ordered_values = qq[0]
    slope, intercept, _ = qq[1]

    # Scatter plot for the data points
    plt.scatter(theoretical_quantiles, ordered_values, color=dot_color, label="Data Points", alpha=0.9)

    # Plot the diagonal line
    plt.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color=line_color, 
             label="Reference Line", linewidth=2)

    # Customize the title and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.6)

    # Add a legend
    plt.legend(fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_turnout_and_electorate(election_data, title="Turnout and Total Electorate at General Elections", 
                                turnout_color="#e07a5f", electorate_color="#3d405b", background_color="#f4f1de"):
    """
    Plots turnout rates as a bar chart and total electorate as a line chart with dual y-axes.

    Parameters:
    - election_data (pd.DataFrame): Data containing 'election', 'turnout_rate', and 'total_electorate' columns.
    - title (str): Title of the plot.
    - turnout_color (str): Color of the turnout bar chart.
    - electorate_color (str): Color of the electorate line chart.
    - background_color (str): Background color of the chart.

    Returns:
    - None: Displays the plot.
    """
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Set background color
    ax1.set_facecolor(background_color)

    # Plot the turnout rate as a bar chart
    ax1.bar(election_data['election'], election_data['turnout_rate'], color=turnout_color, label='Turnout Rate (%)')
    ax1.set_title(title, fontsize=16, pad=15)
    ax1.set_xlabel('Election Year', fontsize=12)
    ax1.set_ylabel('Turnout Rate (%)', fontsize=12, color=turnout_color)
    ax1.tick_params(axis='y', labelcolor=turnout_color)

    # Add grid lines
    ax1.grid(axis='y', linestyle='--', color='grey', alpha=0.7)

    # Adjust x-axis ticks
    ax1.set_xticks(range(len(election_data['election'])))
    ax1.set_xticklabels(election_data['election'], rotation=45, fontsize=10)

    # Add a secondary y-axis for total electorate
    ax2 = ax1.twinx()
    ax2.plot(election_data['election'], election_data['total_electorate'], color=electorate_color, 
             marker='o', label='Total Electorate')
    ax2.set_ylabel('Total Electorate', fontsize=12, color=electorate_color)
    ax2.tick_params(axis='y', labelcolor=electorate_color)

    # Adding a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize=10)

    # Adding a note
    plt.figtext(0.5, -0.03, "Note: 1974F and 1974O are the two general elections held February and October in 1974", 
                ha='center', fontsize=10, wrap=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_contention_distribution(data, contention_column, majority_column, valid_votes_column, 
                                  bins=20, kde=True, color="skyblue", 
                                  percentiles=[0.33, 0.67], 
                                  title="Distribution of Contentiousness", 
                                  xlabel="Contentiousness", ylabel="Frequency"):
    """
    Plots the distribution of contentiousness with KDE and highlights specified percentiles.

    Parameters:
    - data (pd.DataFrame): The dataset containing election results.
    - contention_column (str): Column name to store the calculated contentiousness.
    - majority_column (str): Column name for majority votes.
    - valid_votes_column (str): Column name for valid votes.
    - bins (int): Number of bins for the histogram.
    - kde (bool): Whether to include a KDE plot.
    - color (str): Color of the histogram.
    - percentiles (list of floats): Percentiles to highlight (e.g., [0.33, 0.67]).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None: Displays the plot.
    """
    # Calculate contentiousness
    data[contention_column] = 1 - data[majority_column] / data[valid_votes_column]

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data[contention_column],
        bins=bins,
        kde=kde,
        color=color,
        edgecolor="black",
        alpha=0.8
    )

    # Calculate and add percentile lines
    percentile_values = data[contention_column].quantile(percentiles)
    for i, perc in enumerate(percentiles):
        color_line = "red" if i == 0 else "green"
        plt.axvline(percentile_values.iloc[i], color=color_line, linestyle="--", 
                    label=f"{int(perc * 100)}th percentile")

    # Title and labels
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()



def analyze_turnout_by_contention(election_data, contention_column, turnout_column, 
                                  constituency_column, percentiles, 
                                  bins=[0, 0.33, 0.67, 1], 
                                  bin_labels=['Low', 'Medium', 'High'], 
                                  palette="viridis", figsize=(8, 6)):
    """
    Analyze turnout rates by contentiousness levels and perform One-Way ANOVA.

    Parameters:
    - election_data (pd.DataFrame): Dataset containing election results.
    - contention_column (str): Column name for contentiousness data.
    - turnout_column (str): Column name for turnout rates.
    - constituency_column (str): Column name for constituency names.
    - percentiles (pd.Series or list): Percentiles for binning contentiousness.
    - bins (list): Bin edges for grouping contentiousness levels.
    - bin_labels (list): Labels for each contentiousness group.
    - palette (str): Seaborn palette for boxplot.
    - figsize (tuple): Size of the figure.

    Returns:
    - None: Displays the boxplot and prints ANOVA results.
    """
    # Create a copy of relevant data
    contention_data = election_data[[constituency_column, contention_column, turnout_column]].copy()

    # Categorize contentiousness into bins
    contention_data['group'] = pd.cut(contention_data[contention_column], bins=bins, labels=bin_labels)
    
    anova_result = stats.f_oneway(
        contention_data.loc[contention_data['group'] == 'High', turnout_column],
        contention_data.loc[contention_data['group'] == 'Medium', turnout_column],
        contention_data.loc[contention_data['group'] == 'Low', turnout_column]
    )
    print('ANOVA p-value:', anova_result.pvalue)
    if anova_result.pvalue < 0.05:
        print('The differences in Turnout Rate between groups are statistically significant.')
    else:
        print('The differences in Turnout Rate between groups are not statistically significant.')
    
    # Plot the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(
        x='group',
        y=turnout_column,
        data=contention_data,
        hue='group',
        palette=palette,
    )

    # annotate the anova p-value on the plot, as a bbox on the top right corner
    plt.figtext(0.96, 0.9, f"ANOVA p-value: {anova_result.pvalue:.2e}", ha='right', va='top', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.title('Turnout Rates by Contentiousness Group', fontsize=16, fontweight='bold')
    plt.xlabel('Contentiousness Group', fontsize=14)
    plt.ylabel('Turnout Rate (%)', fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_turnout_by_party(election_data, party_column, turnout_column, party_order=None, palette="viridis", figsize=(10, 6)):
    """
    Analyze turnout rates by winning party and perform One-Way ANOVA.

    Parameters:
    - election_data (pd.DataFrame): Dataset containing election results.
    - party_column (str): Column name for winning party data.
    - turnout_column (str): Column name for turnout rates.
    - party_order (list): Order of parties for the boxplot.
    - palette (str): Seaborn palette for boxplot.
    - figsize (tuple): Size of the figure.

    Returns:
    - None: Displays the boxplot and prints ANOVA results.
    """
    # Categorize "Winning_party" as specified
    election_data["Winning_party"] = election_data[party_column].apply(
        lambda x: "Other" if x not in ["Con", "Lab", "LD"] else x
    )
    
    # Determine party groups
    parties = election_data["Winning_party"].unique()
    if party_order is None:
        party_order = ["Con", "Lab", "LD", "Other"]
    
    # Perform ANOVA
    turnout_by_party = [
        election_data.loc[election_data["Winning_party"] == party, turnout_column]
        for party in party_order if party in parties
    ]
    anova_result = stats.f_oneway(*turnout_by_party)
    print("ANOVA p-value:", anova_result.pvalue)
    if anova_result.pvalue < 0.05:
        print("The differences in Turnout Rate between parties are statistically significant.")
    else:
        print("The differences in Turnout Rate between parties are not statistically significant.")
    
    # Plot the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(
        x="Winning_party",
        y=turnout_column,
        data=election_data,
        order=party_order,
        hue = "Winning_party",
        palette=palette
    )

    # Annotate the ANOVA p-value on the plot
    plt.figtext(
        0.96, 0.9, f"ANOVA p-value: {anova_result.pvalue:.2e}", ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=1")
    )
    
    # Add title and labels
    plt.title("Turnout Rate by Winning Party", fontsize=16, fontweight="bold")
    plt.xlabel("Winning Party", fontsize=14)
    plt.ylabel("Turnout Rate (%)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_turnout_comparison(regional_turnout_2024, regional_turnout_2019, xlim=(50, 75), dot_size=100):
    """
    Plot a comparison of turnout rates for 2019 and 2024 general elections by region.

    Parameters:
        regional_turnout_2024 (pd.DataFrame): DataFrame with regions as index and 2024 turnout rates.
        regional_turnout_2019 (pd.DataFrame): DataFrame with regions as index and 2019 turnout rates.
        xlim (tuple): Tuple specifying the limits of the x-axis (default: (50, 75)).
        dot_size (int): Size of the scatter plot dots (default: 100).
    """
    # Prepare data
    regions = regional_turnout_2024.index  # List of regions
    turnout_2024 = regional_turnout_2024['turnout_rate']
    turnout_2019 = regional_turnout_2019['turnout_rate']

    # Sorting data for consistent ordering
    sorted_indices = turnout_2024.argsort()
    regions = regions[sorted_indices]
    turnout_2024 = turnout_2024.iloc[sorted_indices]
    turnout_2019 = turnout_2019.iloc[sorted_indices]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bars connecting scatter points
    for i, region in enumerate(regions):
        # use dotted lines to connect the points
        ax.plot([turnout_2019.iloc[i], turnout_2024.iloc[i]], [region, region], color='#cdb4db', linewidth=1, linestyle='--')


    # Scatter points for 2019
    ax.scatter(turnout_2019, regions, color='#a2d2ff', label='2019 general election', zorder=3, s=dot_size)

    # Scatter points for 2024
    ax.scatter(turnout_2024, regions, color='#ffc8dd', label='2024 general election', zorder=3, s=dot_size)

    # Remove unwanted spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add labels, title, and legend
    ax.set_xlabel("Turnout Rate (%)")
    ax.set_ylabel("Region")
    ax.set_title("Turnout by Region and Country (2019 vs 2024)")
    ax.legend()

    # Add grid and style adjustments
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(xlim)

    # set background color
    ax.set_facecolor('#f5ebe0')

    # Show plot
    plt.tight_layout()
    plt.show()