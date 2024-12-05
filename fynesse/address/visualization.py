import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import learning_curve
from joblib import load




def plot_regression_results(y_test, y_pred, residuals, title_prefix):
    """
    Generalized function to plot regression results.
    
    Parameters:
        y_test (array-like): True values for the test set.
        y_pred (array-like): Predicted values from the model.
        residuals (array-like): Residuals from the model.
        title_prefix (str): Prefix for the plot titles to distinguish between transformations.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Residual Plot
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title(f"{title_prefix} Residual Plot")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].grid(alpha=0.3)

    # Histogram/KDE of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="blue")
    axes[1].set_title(f"{title_prefix} Histogram/KDE of Residuals")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    # Actual vs. Predicted Plot
    axes[2].scatter(y_test, y_pred, alpha=0.6)
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    axes[2].set_title(f"{title_prefix} Actual vs Predicted")
    axes[2].set_xlabel("Actual Values")
    axes[2].set_ylabel("Predicted Values")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_transformations(y, transformations, titles, figsize=(15, 10), bins=50):
    """
    Generalized function to plot the original and transformed distributions.
    
    Parameters:
        y (array-like): Original target variable.
        transformations (dict): Dictionary of transformations with keys as transformation names
                                and values as functions to apply to `y`.
        titles (list): Titles for each subplot. The first title should correspond to the original data.
        figsize (tuple): Size of the entire figure (default: (15, 10)).
        bins (int): Number of bins for histograms (default: 50).
    """
    # Create a grid for subplots
    num_transformations = len(transformations)
    rows = (num_transformations + 1) // 3 + 1  # Add 1 row for the original distribution
    cols = min(num_transformations + 1, 3)  # At most 3 columns
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot the original distribution
    axes[0].hist(y, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(titles[0])

    # Apply each transformation and plot
    for i, (name, transform) in enumerate(transformations.items(), start=1):
        y_transformed = transform(y)
        axes[i].hist(y_transformed, bins=bins, color=np.random.rand(3,), edgecolor='black', alpha=0.7)
        axes[i].set_title(titles[i])

    # Turn off any extra axes
    for i in range(num_transformations + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()



def plot_regression_diagnostics(
    y_train, y_test, y_pred_train, y_pred_test,
    residuals_train, residuals_test,
    train_sizes, train_scores_mean, test_scores_mean,
    feature_names, coefficients
):
    """
    Plot diagnostics for regression model performance.

    Parameters:
    - y_train: Series or array, actual target values for training data.
    - y_test: Series or array, actual target values for testing data.
    - y_pred_train: Series or array, predicted target values for training data.
    - y_pred_test: Series or array, predicted target values for testing data.
    - residuals_train: Series or array, residuals for training data.
    - residuals_test: Series or array, residuals for testing data.
    - train_sizes: Array of training set sizes for learning curve.
    - train_scores_mean: Array of mean training scores for learning curve.
    - test_scores_mean: Array of mean test scores for learning curve.
    - feature_names: List of feature names.
    - coefficients: Array of feature coefficients.
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.tight_layout(pad=6.0)

    # 1. Residual Plot
    axs[0, 0].scatter(y_pred_train, residuals_train, alpha=0.7)
    axs[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axs[0, 0].set_title("Residual Plot (Training Data)")
    axs[0, 0].set_xlabel("Predicted values")
    axs[0, 0].set_ylabel("Residuals")

    # 2. Residual Distribution Plot
    sns.histplot(residuals_train, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title("Residual Distribution Plot")
    axs[0, 1].set_xlabel("Residuals")

    # 3. Y_real vs. Y_pred
    axs[1, 0].scatter(y_test, y_pred_test, alpha=0.7)
    axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   color="red", linestyle="--", linewidth=1)
    axs[1, 0].set_title("Y_real vs. Y_pred")
    axs[1, 0].set_xlabel("Actual Values")
    axs[1, 0].set_ylabel("Predicted Values")

    # 4. Q-Q Plot
    sm.qqplot(residuals_train, line='45', ax=axs[1, 1])
    axs[1, 1].set_title("Q-Q Plot (Training Residuals)")

    # 5. Learning Curve
    axs[2, 0].plot(train_sizes, train_scores_mean, label="Training score")
    axs[2, 0].plot(train_sizes, test_scores_mean, label="Cross-validation score")
    axs[2, 0].set_title("Learning Curve")
    axs[2, 0].set_xlabel("Training set size")
    axs[2, 0].set_ylabel("R^2 score")
    axs[2, 0].legend(loc="best")

    # 6. Feature Coefficients Barchart
    axs[2, 1].barh(feature_names, coefficients)
    axs[2, 1].set_title("Feature Coefficients")
    axs[2, 1].set_xlabel("Coefficient Value")

    # Display plots
    plt.show()


def plot_model_results(alphas, ridge_scores, lasso_scores, ridge_model, lasso_model, X_columns, best_ridge_alpha, best_lasso_alpha):
    """
    Plot the R^2 scores and feature coefficients for Ridge and Lasso models.

    Parameters:
        - alphas: List of alpha values
        - ridge_scores, lasso_scores: R^2 scores for Ridge and Lasso
        - ridge_model, lasso_model: Best Ridge and Lasso models
        - X_columns: List of feature names
        - best_ridge_alpha, best_lasso_alpha: Best alpha values for Ridge and Lasso
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout(pad=6.0)

    # Ridge Alpha vs. R^2
    axs[0, 0].plot(alphas, ridge_scores, marker='o')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("Ridge Regression: Alpha vs R^2")
    axs[0, 0].set_xlabel("Alpha")
    axs[0, 0].set_ylabel("R^2 Score")
    axs[0, 0].axvline(best_ridge_alpha, color='red', linestyle='--', label=f"Best Alpha: {best_ridge_alpha:.3e}")
    axs[0, 0].legend()

    # Lasso Alpha vs. R^2
    axs[0, 1].plot(alphas, lasso_scores, marker='o')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title("Lasso Regression: Alpha vs R^2")
    axs[0, 1].set_xlabel("Alpha")
    axs[0, 1].set_ylabel("R^2 Score")
    axs[0, 1].axvline(best_lasso_alpha, color='red', linestyle='--', label=f"Best Alpha: {best_lasso_alpha:.3e}")
    axs[0, 1].legend()

    # Feature Coefficients for Best Ridge Model
    axs[1, 0].barh(X_columns, ridge_model.coef_)
    axs[1, 0].set_title(f"Feature Coefficients (Best Ridge, Alpha={best_ridge_alpha:.3e})")
    axs[1, 0].set_xlabel("Coefficient Value")

    # Feature Coefficients for Best Lasso Model
    axs[1, 1].barh(X_columns, lasso_model.coef_)
    axs[1, 1].set_title(f"Feature Coefficients (Best Lasso, Alpha={best_lasso_alpha:.3e})")
    axs[1, 1].set_xlabel("Coefficient Value")

    plt.show()


def plot_residuals(model, data, target):
    """
    Plot residuals for a regression model.

    Parameters:
        model (sklearn.linear_model): The trained regression model.
        data (pd.DataFrame): The input data.
        target (str): The target column to predict.
    """
    X = data.drop(target, axis=1)
    y = data[target]

    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid()
    plt.show()

def plot_learning_curve_subplots(models, data_dict, target, cv=5, scoring='r2'):
    """
    Plot learning curves for all years in subplots.

    Parameters:
        models (dict): Dictionary of trained models.
        data_dict (dict): Dictionary of data for each year.
        target (str): The target column to predict.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric for cross-validation.
    """
    n_years = len(data_dict)
    fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

    if n_years == 1:
        axes = [axes]

    for ax, (year, model_data) in zip(axes, models.items()):
        data = data_dict[year]
        X = data.drop(target, axis=1)
        y = data[target]

        train_sizes, train_scores, test_scores = learning_curve(
            model_data['model'], X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        ax.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
        ax.set_title(f"Learning Curve: {year}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)
        ax.legend(loc="best")
        ax.grid()

    plt.tight_layout()
    plt.show()


def plot_residual_distributions(models, data_dict, target):
    """
    Plot residual distributions for all years in subplots with mean, SD, skewness, and kurtosis.

    Parameters:
        models (dict): Dictionary of trained models.
        data_dict (dict): Dictionary of data for each year.
        target (str): The target column to predict.
    """
    n_years = len(data_dict)
    n_cols = n_years  # Up to 4 columns per row
    n_rows = -(-n_years // n_cols)  # Calculate rows needed (ceiling division)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)

    # Flatten axes array if multiple rows/columns
    axes = axes.flatten() if n_years > 1 else [axes]

    for ax, (year, model_data) in zip(axes, models.items()):
        data = data_dict[year]
        X = data.drop(target, axis=1)
        y = data[target]

        y_pred = model_data['model'].predict(X)
        residuals = y - y_pred

        # Calculate statistics
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        skew_residual = skew(residuals)
        kurtosis_residual = kurtosis(residuals)

        # Plot the residual distribution
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title(f"Residuals Distribution: {year}")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.grid()

        # Add statistics as text
        stats_text = (
            f"Mean: {mean_residual:.2f}\n"
            f"SD: {std_residual:.2f}\n"
            f"Skew: {skew_residual:.2f}\n"
            f"Kurtosis: {kurtosis_residual:.2f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))

    # Hide unused subplots if n_years < n_rows * n_cols
    for ax in axes[len(models):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    """Plot the feature importances of a trained model."""
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances from Random Forest Model")
    plt.gca().invert_yaxis()
    plt.show()
    
    return feature_importances

def plot_paired_feature_importances(feature_importances, features_2021, features_interpolated):
    """Plot a pairwise comparison of 2021 and interpolated feature importances."""
    paired_importances = []
    for f_2021, f_interp in zip(features_2021, features_interpolated):
        importance_2021 = feature_importances.loc[feature_importances["Feature"] == f_2021, "Importance"].values[0]
        importance_interp = feature_importances.loc[feature_importances["Feature"] == f_interp, "Importance"].values[0]
        paired_importances.append({
            "Feature_2021": f_2021, 
            "Importance_2021": importance_2021,
            "Feature_Interpolated": f_interp, 
            "Importance_Interpolated": importance_interp
        })

    paired_df = pd.DataFrame(paired_importances)
    paired_df.plot(kind="bar", x="Feature_2021", y=["Importance_2021", "Importance_Interpolated"], figsize=(12, 8))
    plt.title("Comparison of Feature Importances: 2021 vs Interpolated")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend(["2021 Features", "Interpolated Features"])
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_feature_importance_changes(feature_importance_dict, top_n=10):
    """
    Plot changes in feature importance over the years.

    Parameters:
        feature_importance_dict (dict): Dictionary of feature importances for each year.
        top_n (int): Number of top features to display.
    """
    
    for year in feature_importance_dict.keys():
        feature_importance_dict[year].index = feature_importance_dict[year].index.str[:-5]

    # Combine feature importances into a single DataFrame
    importance_df = pd.DataFrame(feature_importance_dict).T

    # Normalize feature importance values for comparison
    importance_df = importance_df.div(importance_df.sum(axis=1), axis=0)

    # Get the top N features overall
    top_features = importance_df.mean().nlargest(top_n).index

    # Plot feature importance changes for top features
    importance_df[top_features].plot(kind='line', figsize=(12, 8), marker='o')
    plt.title("Feature Importance Changes Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Normalized Importance")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_residuals_heatmap(models, data_dict, target_column='turnout', cmap='coolwarm'):
    """
    Plot a heatmap of residuals for all years.

    Parameters:
        models (dict): A dictionary of trained models for each year.
        data_dict (dict): A dictionary of data for each year.
        target_column (str): The name of the target column in the dataset.
        cmap (str): The colormap to use for the heatmap (default is 'coolwarm').
    """
    # Create a DataFrame of residuals for each year
    residuals_df = pd.DataFrame({
        year: data[target_column] - models[year]['model'].predict(data.drop(target_column, axis=1))
        for year, data in data_dict.items()
    })
    
    # Plot the heatmap
    sns.heatmap(residuals_df, cmap=cmap, center=0, annot=False)
    plt.title("Residuals Heatmap Over Years")
    plt.xlabel("Year")
    plt.ylabel("Samples")
    plt.show()