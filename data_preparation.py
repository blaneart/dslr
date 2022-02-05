import pandas as pd
from pandas.api.types import is_numeric_dtype


house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']


def prepare_dataset(df, house_name):
    df['label'] = df['Hogwarts House'].apply(lambda x: x == house_name).astype(int)
    df = df.drop(['Care of Magical Creatures', 'Arithmancy', 'Astronomy'], axis=1)
    final_df = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1)
    final_df = pd.concat([df['Hogwarts House'], final_df], axis=1)
    x = final_df.drop(['label', 'Hogwarts House'], axis=1)
    x_train = x[x['val'] == 0].drop('val', axis=1).reset_index(drop=True)
    x_val = x[x['val'] == 1].drop('val', axis=1).reset_index(drop=True)
    y_train = final_df[final_df['val'] == 0]['label'].reset_index(drop=True)
    y_val = final_df[final_df['val'] == 1]['label'].reset_index(drop=True)

    return x_train, x_val, y_train, y_val

def get_number_of_classes(df):
    count_of_classes = {}
    for house in house_names:
        count_of_classes[house] = (df.groupby('Hogwarts House').count()['Arithmancy'][house])
    print(count_of_classes)
    return count_of_classes

def train_validation_splitter(df, part=0.9):
    count_of_classes = get_number_of_classes(df)
    df['val'] = -1
    for house in house_names:
        for i, (index, row) in enumerate(df.iterrows()):
            if int(count_of_classes[house] * part) >i:
                df.loc[index, 'val'] = 1
            else:
                df.loc[index, 'val'] = 0
    print(df)
    print(df.groupby(['Hogwarts House', 'val']).count())
    return df

def data_preparation(path, house):
    df = pd.read_csv(path)
    df2 = df.drop(df.columns[df.apply(lambda col: not is_numeric_dtype(col))], axis=1).drop('Index', axis=1)
    df2 = pd.concat([df['Hogwarts House'], df2], axis=1)
    ## Fill Nan values with mean value for each house for each course ###
    for column in df2:
        if column != 'Hogwarts House':
            df2[column].fillna(df2.groupby('Hogwarts House')[column].transform("mean"), inplace=True)
    df2 = train_validation_splitter(df2, part=0.9)
    x_train, x_val, y_train, y_val = prepare_dataset(df2, house)
    return(x_train, x_val, y_train, y_val)
