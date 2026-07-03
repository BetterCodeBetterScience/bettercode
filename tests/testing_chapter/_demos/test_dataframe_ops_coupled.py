import pandas as pd

from bettercode.testing.dataframe_ops import get_initials, split_names

# Shared global DataFrame — coupled/brittle tests demo (pedagogical negative example)
# These tests share a global that creates ordering dependency
people_df = pd.DataFrame({'name': ['Alice Smith', 'Bob Howard', 'Charlie Ashe']})


def test_split_names():
    split_names(people_df)
    assert people_df['firstname'].tolist() == ['Alice', 'Bob', 'Charlie']
    assert people_df['lastname'].tolist() == ['Smith', 'Howard', 'Ashe']


def test_get_initials():
    get_initials(people_df)
    assert people_df['initials'].tolist() == ['AS', 'BH', 'CA']
