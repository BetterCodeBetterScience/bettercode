
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compute_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation matrix for a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numerical columns

    Returns
    -------
    pd.DataFrame
        Spearman correlation matrix
    """
    return df.corr(method="spearman")


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlation matrix using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numerical columns
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'spearman')

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return df.corr(method=method)


def filter_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a dataframe to keep only numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with only numerical columns
    """
    numerical_df = df.select_dtypes(include=["number"])
    return numerical_df


def filter_meaningful_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Filter meaningful variables dataframe to numerical columns only.

    Parameters
    ----------
    df : pd.DataFrame
        Meaningful variables dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only numerical columns
    """
    return filter_numerical_columns(df)


def filter_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Filter demographics dataframe to numerical columns only.

    Parameters
    ----------
    df : pd.DataFrame
        Demographics dataframe

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only numerical columns
    """
    return filter_numerical_columns(df)


def join_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Join two dataframes based on their index.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
    how : str
        Type of join: 'inner', 'outer', 'left', 'right' (default: 'inner')

    Returns
    -------
    pd.DataFrame
        Joined dataframe
    """
    return df1.join(df2, how=how, lsuffix="_mv", rsuffix="_demo")


def join_meaningful_and_demographics(
    meaningful_vars: pd.DataFrame,
    demographics: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Join meaningful variables and demographics dataframes.

    Parameters
    ----------
    meaningful_vars : pd.DataFrame
        Meaningful variables dataframe (filtered to numerical)
    demographics : pd.DataFrame
        Demographics dataframe (filtered to numerical)
    how : str
        Type of join (default: 'inner')

    Returns
    -------
    pd.DataFrame
        Joined dataframe
    """
    return join_dataframes(meaningful_vars, demographics, how=how)


def load_csv_from_url(url: str, index_col: int = 0) -> pd.DataFrame:
    """Load a CSV file from a URL.

    Parameters
    ----------
    url : str
        URL to the CSV file
    index_col : int
        Column to use as index (default: 0, first column)

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with the first column as index
    """
    return pd.read_csv(url, index_col=index_col)


def load_meaningful_variables(
    url: str = "https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/meaningful_variables_clean.csv",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Load the meaningful variables dataset.

    Parameters
    ----------
    url : str
        URL to the meaningful variables CSV file
    cache_path : Path, optional
        If provided, save/load from this local path

    Returns
    -------
    pd.DataFrame
        Meaningful variables dataframe
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0)

    df = load_csv_from_url(url)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df


def load_demographics(
    url: str = "https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/demographics.csv",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Load the demographics dataset.

    Parameters
    ----------
    url : str
        URL to the demographics CSV file
    cache_path : Path, optional
        If provided, save/load from this local path

    Returns
    -------
    pd.DataFrame
        Demographics dataframe
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0)

    df = load_csv_from_url(url)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df


def generate_clustered_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 10),
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> sns.matrix.ClusterGrid:
    """Generate a clustered heatmap from a correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : Path, optional
        If provided, save the figure to this path
    figsize : tuple
        Figure size (width, height) in inches
    cmap : str
        Colormap name (default: 'coolwarm')
    vmin : float
        Minimum value for color scale
    vmax : float
        Maximum value for color scale

    Returns
    -------
    sns.matrix.ClusterGrid
        The ClusterGrid object containing the heatmap
    """
    # Create clustered heatmap
    g = sns.clustermap(
        corr_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=False,
        yticklabels=True,
    )

    # Set y-axis label font size
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=3)

    # Set title
    g.fig.suptitle("Clustered Correlation Heatmap (Spearman)", y=1.02, fontsize=14)

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(output_path, dpi=300, bbox_inches="tight")

    return g


def save_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a correlation matrix to a CSV file.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : Path
        Path to save the CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(output_path)
