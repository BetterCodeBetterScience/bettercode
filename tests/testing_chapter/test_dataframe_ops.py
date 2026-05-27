import pandas as pd

from bettercode.testing.dataframe_ops import get_initials, split_names


def get_people_df():
    return pd.DataFrame({'name': ['Alice Smith', 'Bob Howard', 'Charlie Ashe']})


def test_split_names_fullsetup():
    local_people_df = get_people_df()
    split_names(local_people_df)
    assert local_people_df['firstname'].tolist() == ['Alice', 'Bob', 'Charlie']
    assert local_people_df['lastname'].tolist() == ['Smith', 'Howard', 'Ashe']


def test_get_initials_fullsetup():
    local_people_df = get_people_df()
    split_names(local_people_df)
    get_initials(local_people_df)
    assert local_people_df['initials'].tolist() == ['AS', 'BH', 'CA']
