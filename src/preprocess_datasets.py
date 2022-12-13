import os
import numpy as np
import pandas as pd
import pickle
import sqlite3
from bs4 import BeautifulSoup

# folders
DATA_FOLDER = "data"
INPUT_FOLDER = os.path.join(DATA_FOLDER, "external")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "processed")
EMBEDDINGS_FOLDER = os.path.join(INPUT_FOLDER, "embeddings", "ger-all_sgns")
CONCRETENESS_FOLDER = os.path.join(INPUT_FOLDER, "affective_norms")
MILLION_POSTS_FOLDER = os.path.join(INPUT_FOLDER, "million_post_corpus")

# filenames
CORPUSDB = "corpus.sqlite3"
CONCRETENESS_FILE = "affective_norms.txt"


def process_ger_concreteness(folder = CONCRETENESS_FOLDER, filename = CONCRETENESS_FILE, output = OUTPUT_FOLDER, all_lower_case = True):
    """
    This function processes a German concreteness ratings file, converts the words to lower case (if specified), and saves the processed file as a pickle.

    Inputs:
        folder: the directory where the concreteness ratings file is stored
        filename: the name of the concreteness ratings file
        output: the directory where the processed file will be saved
        all_lower_case: a boolean indicating whether to convert all words to lower case
    """
    # create the path to the output file and to the concreteness ratings file
    output_file = os.path.join(OUTPUT_FOLDER, "concreteness_ger.pkl")
    concreteness_file = os.path.join(CONCRETENESS_FOLDER, filename)
    # read in the concreteness ratings file
    concreteness = pd.read_csv(concreteness_file, sep='\t', index_col='Word')
    if all_lower_case:
        # change all words to lower case, this results in duplicates (eg noun and verb)
        concreteness.index = concreteness.index.str.lower()
    # add concreteness = 10 - abstractness
    concreteness.eval(' `concreteness` = 10 - `AbstConc` ', inplace=True)
    # save the processed file as a pickle
    concreteness.to_pickle(output_file[['concreteness']])


def get_all_text_from_article(row):
    """
    This function joins texts from headline, secondary headline, and paragraphs and returns it as a single string.

    Inputs:
        row: a row of article data containing the article's title, body, and any other relevant information

    Output: a string containing the text of the article
    """
    # extract the article's headline
    headline = row['Title']
    # create a BeautifulSoup object to parse the article body
    soup = BeautifulSoup(row['Body'], features="html.parser")
    # some articles don't contain h2
    try:
        headline2 = soup.find('h2').text
    except:
        headline2 = ""
    
    # get the text from all 'p' tags in the article body
    paragraphs = [paragraph.text for paragraph in soup.find_all('p')]
    
    # join the headline, headline2, and all paragraphs into a single string
    joined_article_text = ' '.join([ headline, headline2, *paragraphs ])

    return joined_article_text.lower()


def process_non_conformity(folder=MILLION_POSTS_FOLDER, database = CORPUSDB, output_folder=OUTPUT_FOLDER):
    """
    This function processes the data from the Posts and Articles tables in the provided Million Posts database,
    joins the text from all posts and articles, and saves the processed data to a pickle file.

    Inputs:
        folder: the directory where the Million Posts database is stored (default MILLION_POSTS_FOLDER)
        database: the name of the Million Posts database file (default CORPUSDB)
    """
    # connect to the database and read the posts and articles tables to dataframes
    database = os.path.join(folder, database)
    with sqlite3.connect(database) as con:
        posts = pd.read_sql_query("SELECT Headline, Body FROM Posts", con)
        articles = pd.read_sql_query("SELECT Title, Body FROM Articles", con)

    # join the post headline and body into a single column
    posts['head_body'] = posts['Headline'].fillna("") + " " + posts['Body'].fillna("")
    # join all posts
    joined_posts = ' '.join(posts["head_body"])
    # convert to lower case and remove newline characters
    joined_posts = joined_posts.lower().replace('\r\n', ' ')
    
    # get all text from each article 
    articles['text'] = articles.apply(lambda row: get_all_text_from_article(row), axis=1)
    # join all articles
    joined_articles = ' '.join(articles['text'])
    
    # save to pickles
    with open(os.path.join(output_folder,'posts.pkl'), 'wb') as posts_pickle:
        pickle.dump(joined_posts, posts_pickle)

    with open(os.path.join(output_folder,'articles.pkl'), 'wb') as articles_pickle:
        pickle.dump(joined_articles, articles_pickle)
    

if __name__ == "__main__":
    process_ger_concreteness()
    process_non_conformity()
