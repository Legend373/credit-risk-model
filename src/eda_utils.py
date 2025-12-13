"""
Reusable EDA utility functions
--------------------------------
Updated for dataset schema with column definitions.
Designed ONLY for exploratory analysis.
Can be safely imported and used inside notebooks/eda.ipynb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# General overview utilities
# ------------------------------------------------------------------

def overview(df: pd.DataFrame) -> None:
    """Print basic structure information about the dataset."""
    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Data Types & Non-Null Counts ---")
    print(df.info())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> None:
    """
    Display improved summary statistics based on dataset columns.
    Separates true numerical, discrete numeric, categorical, and datetime features.
    """

    # Parse datetime column
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

    # Identify columns by type and dataset knowledge
    true_numeric = ['Amount', 'Value']
    discrete_numeric = ['PricingStrategy', 'FraudResult', 'CountryCode']
    categorical = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                   'CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    datetime_cols = ['TransactionStartTime']

    # True numerical
    if true_numeric:
        print("\n--- Numerical Features Summary (True Continuous) ---")
        display(df[true_numeric].describe().T)

    # Discrete numeric
    if discrete_numeric:
        print("\n--- Discrete / Encoded Numerical Features ---")
        display(df[discrete_numeric].agg(['count', 'nunique', 'min', 'max']).T)

    # Categorical
    if categorical:
        print("\n--- Categorical Features Summary ---")
        display(df[categorical].describe().T)

    # Datetime
    if datetime_cols:
        print("\n--- Datetime Features Summary ---")
        display(df[datetime_cols].agg(['count', 'min', 'max']).T)


# ------------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------------

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Plot histogram + boxplot for true numerical columns."""
    num_cols = ['Amount', 'Value']

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")

        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")

        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(df: pd.DataFrame, top_n: int = 10) -> None:
    """Plot bar charts for top categories in categorical columns."""
    cat_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                'CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().head(top_n).plot(kind='bar')
        plt.title(f"Top {top_n} Categories in {col}")
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numerical features including discrete numeric features."""
    num_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult', 'CountryCode']
    corr = df[num_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, annot=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Missing value analysis
# ------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing value counts and percentages."""
    report = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct': (df.isna().mean() * 100).round(2),
    })
    report = report[report['missing_count'] > 0]
    return report.sort_values('missing_pct', ascending=False)


# ------------------------------------------------------------------
# Outlier detection
# ------------------------------------------------------------------

def outlier_boxplots(df: pd.DataFrame) -> None:
    """Visualize outliers using boxplots for true numerical features."""
    num_cols = ['Amount', 'Value']

    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers in {col}")
        plt.tight_layout()
        plt.show()
