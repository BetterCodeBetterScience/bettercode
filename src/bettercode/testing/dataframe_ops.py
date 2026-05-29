import pandas as pd


def split_names(df: pd.DataFrame) -> None:
    """Split each full name into ``firstname`` and ``lastname`` columns in place."""
    df['firstname'] = df['name'].apply(lambda x: x.split()[0])
    df['lastname'] = df['name'].apply(lambda x: x.split()[1])


def get_initials(df: pd.DataFrame) -> None:
    """Add an ``initials`` column from the first and last name initials in place."""
    df['initials'] = df['firstname'].str[0] + df['lastname'].str[0]
