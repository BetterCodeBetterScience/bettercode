"""Script to remove Motivation variables from demographics CSV file.

This script is used in the datalad example to demonstrate data modification.

Usage
-----
    python modify_data.py <path_to_demographics_csv>
"""

import sys
import pandas as pd


def remove_motivation_columns(filepath: str) -> None:
    """Remove all columns containing 'Motivation' from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the demographics CSV file to modify
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('Motivation')]
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python modify_data.py <path_to_demographics_csv>"
    remove_motivation_columns(sys.argv[1])