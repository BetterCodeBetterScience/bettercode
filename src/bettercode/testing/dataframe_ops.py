def split_names(df):
    df['firstname'] = df['name'].apply(lambda x: x.split()[0])
    df['lastname'] = df['name'].apply(lambda x: x.split()[1])


def get_initials(df):
    df['initials'] = df['firstname'].str[0] + df['lastname'].str[0]
