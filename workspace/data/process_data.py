import sys
import csv
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """load data"""
    messages = pd.read_csv("disaster_messages.csv")
    categories = pd.read_csv("disaster_categories.csv")
    df = pd.merge(messages,categories,on = "id")
    return df


def clean_data(df):
    """clean all the data"""
    categories = df["categories"].str.split(";",expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0:1,]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x.str.split("-")[0][0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string.str.split("-")[1][1]
        categories[column] = categories[column].apply(lambda x:x.split("-")[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    del df['categories']
    df = pd.concat([df,categories], axis=1)
    # check number of duplicates
    df[df.duplicated()==True]
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """saving the data"""
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('df_uda', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
