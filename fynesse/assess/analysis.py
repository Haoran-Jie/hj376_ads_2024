import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.interpolate import interp1d, CubicSpline
from tqdm import tqdm
from .preprocessing import remove_duplicate_columns
from ..constants import COLUMN_PREFIXES

def analyze_correlation(features, feature_counts, title, threshold=0.8):
    """
    Analyze feature correlations, plot a heatmap, and identify highly correlated feature pairs.

    Parameters:
        features (list): List of feature names to analyze.
        feature_counts (pd.DataFrame): DataFrame containing feature counts.
        title (str): Title for the heatmap.
        threshold (float): Correlation threshold to identify highly correlated pairs.
    """
    # Subset the DataFrame for the selected features
    revised_feature_counts = feature_counts[features]

    # Calculate the correlation matrix
    correlation_matrix = revised_feature_counts.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title(title)
    plt.show()

    # Identify highly correlated feature pairs
    highly_correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):  # Only look at one side of the matrix
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                highly_correlated_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))

    # Display highly correlated pairs
    print(f"Highly Correlated Feature Pairs for {title}:")
    for pair in highly_correlated_pairs:
        print(f"{pair[0]} and {pair[1]} have a correlation of {pair[2]:.2f}")

    return highly_correlated_pairs

def compute_pairwise_correlations(df, feature_columns, target_column):
    """Compute pairwise correlations of features with a target column."""
    correlations = df[feature_columns + [target_column]].corr()[target_column]
    return correlations.sort_values(ascending=False)


def compute_correlations(df, feature_groups, target_column, area_column, residents_column="total_residents"):

    df_clean = df.loc[(df[feature_groups].sum(axis=1) > 0) & (df[target_column] > 0)]
    correlations = {}
    for group in feature_groups:
        feature_norm = df_clean[group] / df_clean[feature_groups].sum(axis=1)
        feature_density = df_clean[group] / df_clean[area_column]
        feature_population = df_clean[group] / df_clean[residents_column]

        correlation_norm = feature_norm.corr(df_clean[target_column])
        correlation_density = feature_density.corr(df_clean[target_column])
        correlation_population = feature_population.corr(df_clean[target_column])
        correlations[f"{group}_norm"] = correlation_norm
        correlations[f"{group}_density"] = correlation_density
        correlations[f"{group}_population"] = correlation_population
    return pd.Series(correlations).sort_values(ascending=False)


def test_linearity(df, col_prefix, years=[2001, 2011, 2021], threshold=0.7):
    """
    Test the linearity of a variable across multiple years.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col_prefix (str): The column prefix to test for linearity.
        years (list): A list of years to test for linearity.
        threshold (float): The R-squared threshold for linearity.

    Returns:
        str: Either 'linear' or 'cubic' depending on the overall linearity of the column.
    """
    linear_count = 0
    total_count = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Testing Linearity for {col_prefix}"):
        y = [row[f"{col_prefix}_{year}"] for year in years]
        x = years
        # Fit a linear regression
        _, _, r_value, _, _ = linregress(x, y)
        if abs(r_value) >= threshold:
            linear_count += 1
        total_count += 1
    # Decide method based on proportion of linear trends
    return 'linear' if linear_count / total_count > 0.7 else 'cubic'


def interpolate_data(df, columns, years=[2001, 2011, 2021], target_years=[2010, 2015, 2017, 2019, 2024], methods=None):
    """
    Interpolate data for multiple target years using the specified interpolation methods.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column prefixes to interpolate (e.g., ['white_british', 'asian_total']).
        years (list): The years corresponding to the data columns (default: [2001, 2011, 2021]).
        target_years (list): List of years to interpolate (e.g., [2010, 2015, 2017, 2019]).
        methods (dict): Dictionary mapping column prefixes to interpolation methods ('linear' or 'cubic').

    Returns:
        pd.DataFrame: The original DataFrame with new columns containing interpolated values for each target year.
    """
    if methods is None:
        raise ValueError("Methods dictionary must be provided for interpolation.")

    interpolated_results = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Interpolating Data"):
        row_result = {'OA11CD': row['OA11CD']}  # Identifier for each OA

        for col_prefix in columns:
            # Extract the values for interpolation
            data_values = [row[f"{col_prefix}_{year}"] for year in years]

            # Check for missing values
            if any(pd.isnull(data_values)):
                for target_year in target_years:
                    row_result[f"{col_prefix}_{target_year}"] = np.nan  # Cannot interpolate with missing data
            else:
                # Determine interpolation method
                method = methods.get(col_prefix, 'linear')  # Default to 'linear' if not specified

                # Perform interpolation
                if method == 'linear':
                    interpolator = interp1d(years, data_values, kind='linear', fill_value="extrapolate")
                elif method == 'cubic':
                    interpolator = CubicSpline(years, data_values)
                else:
                    raise ValueError(f"Invalid method for {col_prefix}. Use 'linear' or 'cubic'.")

                # Interpolate for all target years
                for target_year in target_years:
                    row_result[f"{col_prefix}_{target_year}"] = interpolator(target_year)

        interpolated_results.append(row_result)

    # Convert results to a DataFrame
    interpolated_df = pd.DataFrame(interpolated_results)

    # Merge the interpolated values back to the original DataFrame
    result_df = pd.merge(df, interpolated_df, on='OA11CD')
    return result_df


def process_interpolation_oa(df_direct, df_approx, columns, years=[2001, 2011, 2021], target_years=[2010, 2015, 2017, 2019, 2024], threshold=0.7):
    """
    Full pipeline to determine interpolation methods using df_direct and perform interpolation on df_approx.

    Parameters:
        df_direct (pd.DataFrame): DataFrame for direct matching (unchanged OAs).
        df_approx (pd.DataFrame): DataFrame for approximated matching (aggregated OAs).
        columns (list): List of column prefixes to process (e.g., ['white_british', 'asian_total']).
        years (list): The years corresponding to the data columns (default: [2001, 2011, 2021]).
        target_years (list): List of years to interpolate (e.g., [2010, 2015, 2017, 2019]).
        threshold (float): The R-squared threshold for linearity.

    Returns:
        pd.DataFrame: The DataFrame with interpolated values for target years.
    """
    methods = {}

    # Step 1: Determine interpolation methods using df_direct
    for col_prefix in tqdm(columns, desc="Determining Interpolation Methods"):
        methods[col_prefix] = test_linearity(df_direct, col_prefix, years=years, threshold=threshold)
        print(f"{col_prefix}: {methods[col_prefix]} interpolation chosen")

    # Step 2: Perform interpolation using df_approx
    interpolated_df = interpolate_data(df_approx, columns, years=years, target_years=target_years, methods=methods)

    return interpolated_df

def plot_boxplot(data, x_col, y_col, title, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=x_col,
        y=y_col,
        data=data,
        hue = x_col,
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def merge_total_population(interpolated_df, total_df, total_col_name):
    """
    Merge total population data with the interpolated DataFrame.

    Parameters:
        interpolated_df (pd.DataFrame): Interpolated data.
        total_df (pd.DataFrame): DataFrame containing total population values.
        total_col_name (str): Column name for the total population.

    Returns:
        pd.DataFrame: Updated DataFrame with total population merged.
    """
    return interpolated_df.merge(
        total_df[['geography_code', total_col_name]], 
        left_on='OA11CD', 
        right_on='geography_code', 
        how='left'
    )

def calculate_counts(interpolated_df, columns, years, total_col_name):
    """
    Calculate counts from proportions using total population.

    Parameters:
        interpolated_df (pd.DataFrame): Interpolated data.
        columns (list): Column prefixes to process.
        years (list): Target years.
        total_col_name (str): Column name for the total population.

    Returns:
        pd.DataFrame: Updated DataFrame with counts calculated.
    """
    for col in columns:
        for year in years:
            interpolated_df[f"{col}_{year}_count"] = (
                interpolated_df[f"{col}_{year}"] * interpolated_df[total_col_name]
            )
    return interpolated_df

def aggregate_to_pcon(interpolated_df, columns, years, total_col_name, pcon_map, pcon_col='PCON11CD'):
    """
    Aggregate counts and compute proportions by parliamentary constituency.

    Parameters:
        interpolated_df (pd.DataFrame): Data with counts calculated.
        columns (list): Column prefixes to process.
        years (list): Target years.
        total_col_name (str): Column name for the total population.
        pcon_map (pd.DataFrame): Mapping of OA to PCON.

    Returns:
        pd.DataFrame: Aggregated data by PCON.
    """
    # Merge OA to PCON mapping
    interpolated_df = interpolated_df.merge(pcon_map, on='OA11CD', how='left')

    # Group by PCON and aggregate counts
    grouped_df = interpolated_df.groupby(pcon_col).agg(
        {
            total_col_name: 'sum',
            **{f"{col}_{year}_count": 'sum' for col in columns for year in years}
        }
    ).reset_index()

    # Compute proportions
    for col in columns:
        for year in years:
            grouped_df[f"frac_{col}_{year}"] = (
                grouped_df[f"{col}_{year}_count"] / grouped_df[total_col_name]
            )
    return grouped_df

def process_pipeline_pcon(result_dfs, total_dfs, pcon_map, config, years=[2010, 2015, 2017, 2019], pcon_col='PCON11CD'):
    """
    Process all pipelines (ethnic group, household composition, deprivation, qualification).

    Parameters:
        result_dfs (dict): Dictionary of interpolated result DataFrames for each pipeline.
        total_dfs (dict): Dictionary of total population DataFrames for each pipeline.
        pcon_map (pd.DataFrame): Mapping of OA11CD to PCON11CD.
        config (dict): Configuration dictionary with column prefixes and total column names.
        years (list): Target years for interpolation.

    Returns:
        dict: Dictionary of processed DataFrames for each pipeline aggregated by PCON.
    """
    final_results = {}

    for pipeline, params in config.items():
        print(f"Processing {pipeline} pipeline...")

        # Step 1: Copy interpolated data
        interpolated_df = result_dfs[pipeline].copy()

        # Step 2: Merge total population
        interpolated_df = merge_total_population(
            interpolated_df, 
            total_dfs[pipeline], 
            params['total_col']
        )

        # Step 3: Calculate counts
        interpolated_df = calculate_counts(
            interpolated_df, 
            params['columns'], 
            years, 
            params['total_col']
        )

        # Step 4: Aggregate to PCON and compute proportions
        final_results[pipeline] = aggregate_to_pcon(
            interpolated_df, 
            params['columns'], 
            years, 
            params['total_col'], 
            pcon_map,
            pcon_col=pcon_col
        )
    
    print("___ Processing complete. ___")
    
    return final_results

def get_merged_census_results(final_results):
    # Access results for individual pipelines
    ethnic_group_results = final_results['ethnic_group']
    household_results = final_results['household_composition']
    deprivation_results = final_results['deprivation']
    qualification_results = final_results['qualification']
    economic_activity_results = final_results['economic_activity']

    merged_census_results = pd.concat([ethnic_group_results, household_results, deprivation_results, qualification_results, economic_activity_results], axis=1)
    merged_census_results = remove_duplicate_columns(merged_census_results, "interpolated_census_result")
    return merged_census_results

def get_yearly_census_results(merged_census_results, election_results, years=[2010, 2015, 2017, 2019]):
    M = {}
    for year in years:
        election_results_history_now = election_results[election_results['election'] == year]
        election_results_history_merged = election_results_history_now.merge(merged_census_results, left_on='constituency_id', right_on='PCON11CD', how='inner')
        # sort by constituency_id
        election_results_history_merged = election_results_history_merged.sort_values('constituency_id')
        feature_columns = []
        for key, cols in COLUMN_PREFIXES.items():
            for col in cols:
                feature_columns.append(f"frac_{col}_{year}")
        M[year] = election_results_history_merged[feature_columns + ['turnout']]
    
    return M