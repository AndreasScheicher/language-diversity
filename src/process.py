"""Preprocess all necessary data sets for the project."""

import os
import sqlite3

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import functions

# define folders
DATA_FOLDER = "data"
INPUT_FOLDER = os.path.join(DATA_FOLDER, "external")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "processed")
EMBEDDINGS_FOLDER = os.path.join(INPUT_FOLDER, "embeddings", "ger-all_sgns")
CONCRETENESS_FOLDER = os.path.join(INPUT_FOLDER, "affective_norms")
CONCRETENESS_FOLDER_ENG = os.path.join(INPUT_FOLDER, "concreteness_ratings_eng")
MILLION_POSTS_FOLDER = os.path.join(INPUT_FOLDER, "million_post_corpus")

# define filenames
CORPUSDB = "corpus.sqlite3"
CONCRETENESS_FILE = "affective_norms.txt"
CONCRETENESS_FILE_ENG = "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
CONCRETENESS_PICKLE_ENG = "concreteness_eng.pkl"


def process_ger_concreteness(
    folder = CONCRETENESS_FOLDER, filename = CONCRETENESS_FILE,
    output_folder = OUTPUT_FOLDER, all_lower_case = True):
    """Process the German concreteness ratings file and store to pickle.

    The processing includes reading the csv, changing all words to
    lowercase if specified, and renaming the concreteness column.

    Parameters:
    folder (str): where the concreteness ratings file is stored.
    filename (str): the name of the concreteness ratings file.
    output_folder (str): where the processed file will be saved.
    all_lower_case (boolean): if true, convert all words to lower case.
    """
    # create the path to the input and output files
    output_file = os.path.join(output_folder, "concreteness_ger.pkl")
    concreteness_file = os.path.join(folder, filename)
    # read in the concreteness ratings file
    concreteness = pd.read_csv(concreteness_file, sep='\t', index_col='Word')
    if all_lower_case:
        # change all words to lower case
        # this results in duplicates (eg noun and verb)
        concreteness.index = concreteness.index.str.lower()
    # rename column to concreteness
    concreteness.rename(columns={'AbstConc': 'concreteness'}, inplace=True)
    # save the processed file as a pickle
    concreteness[['concreteness']].to_pickle(output_file)


def get_all_text_from_article(row):
    """Join the text from headline, secondary headline, and paragraphs
    and return it as a single string.

    Parameters:
    row: a row of article data containing the article's title, body,
    and any other information

    Returns: a string containing the text of the article
    """
    # extract the article's headline
    headline = row['Title']
    # create a BeautifulSoup object to parse the article body
    soup = BeautifulSoup(row['Body'], features="html.parser")
    # extract headline2 if the article has it
    try:
        headline2 = soup.find('h2').text
    except AttributeError:
        headline2 = ""

    # get the text from all 'p' tags in the article body
    paragraphs = [paragraph.text for paragraph in soup.find_all('p')]

    # join the headline, headline2, and all paragraphs into one string
    joined_article_text = ' '.join([ headline, headline2, *paragraphs ])

    return joined_article_text.lower()


def process_non_conformity(
        folder=MILLION_POSTS_FOLDER, database = CORPUSDB,
        output_folder=OUTPUT_FOLDER):
    """Process Posts and Articles from the Million Posts database,
    and get the ratio of word frequencies in posts and articles.

    This ratio is supposed to be used as a proxy for the word use in
    non-conform contexts.

    Parameters:
    folder (str): the directory of the Million Posts database
    database (str): the name of the Million Posts database
    output_folder (str): the directory where the output pickle file
    will be saved
    """
    # get the filepath for the output
    output_file = os.path.join(output_folder, "non-conformity_ger.pkl")

    # connect to the database
    database = os.path.join(folder, database)
    with sqlite3.connect(database) as con:
        # read the posts and articles tables to pandas dataframes
        posts = pd.read_sql_query("SELECT Headline, Body FROM Posts", con)
        articles = pd.read_sql_query("SELECT Title, Body FROM Articles", con)

    # fill na entries with an empty string
    posts['Headline'].fillna("", inplace=True)
    posts['Body'].fillna("", inplace=True)
    # join the post headline and body into a single column
    posts['head_body'] = posts['Headline'] + " " + posts['Body']
    # join all posts
    joined_posts = ' '.join(posts["head_body"])
    # convert to lower case and remove newline characters with space
    joined_posts = joined_posts.lower().replace('\r\n', ' ')
    # get all text from each article
    articles['text'] = articles.apply(get_all_text_from_article, axis=1)
    # join all articles into one string
    joined_articles = ' '.join(articles['text'])
    # use the words from the embeddings as vocabulary
    vocabulary = functions.get_vocab_from_embeddings()
    # get word frequencies of posts and articles
    texts = [joined_posts, joined_articles]
    count_vec = CountVectorizer(vocabulary=vocabulary)
    count_matrix = count_vec.fit_transform(texts)
    # scipy csr matrix to numpy array
    freq = count_matrix.todense()
    # get log of freqency ratio as non-conformity measure
    non_conf = pd.DataFrame(
        np.array(np.log(freq[0] / freq[1]))[0],
        index=vocabulary, columns=['non-conformity'])
    # drop na and inf
    with pd.option_context('mode.use_inf_as_na', True):
        non_conf.dropna(inplace=True)
    # save to pickle
    non_conf.to_pickle(output_file)


def get_polysemy_score_evolution(
        start_year = 1950, end_year = 1990, folder=OUTPUT_FOLDER,
        language = 'ger', vocabulary = None, verbose = False,
        percentile=90):
    """Calculate the clustering coefficients for words in a given range
    of years, and save the results to a pickle.

    Performs the following steps for each decade in the specified range:
    1. Load the matrix and vocabulary for the current decade.
    2. Remove empty words from the matrix and vocabulary.
    3. Calculate the cosine similarity for the reduced matrix.
    4. Calculate the clustering coefficient for the cosine similarity.
    5. Store the results in a dataframe and save to pickle.

    Parameters:
    start_year (int): The starting year (inclusive)
    end_year (int): The ending year (inclusive)
    folder (str): The folder where the output file will be saved.
    language (str): lowercase 3 character code of language
    vocabulaty (list): list of vocabulary for word count
    verbose (bool): If True, print status messages
    percentile (int): only connections above percentile will be used.
    """
    # define the output file path
    output_file = os.path.join(folder, f'polysemy_score_years_{language}.pkl')
    # define names of embeddings files
    embeddings_list = {
        "eng": os.path.join("eng-all_sgns", "sgns"),
        "fra": "fre-all_sgns",
        "ger": "ger-all_sgns"
    }
    # iterate decades
    for year in range(start_year, end_year+10, 10):
        if verbose:
            print("Year:", year)
        # Load the matrix and vocabulary for the current year
        mat = functions.load_mat(year,
                                 language_folder=embeddings_list[language])
        vocab = functions.load_vocab(year,
                                     language_folder=embeddings_list[language])

        if verbose:
            print("Ratio of non-zero vectors:", functions.check_sparcity(mat))
        # Remove zero vectors from matrix
        reduced_mat, reduced_vocab = functions.remove_empty_words(mat, vocab)
        # if vocabulary is passed, remove all other words
        if vocabulary is not None:
            reduced_mat, reduced_vocab = functions.reduce_to_list_of_words(
                reduced_mat, reduced_vocab, vocabulary)
        # Calculate the cosine similarity between all word-vectors
        cos_sim = cosine_similarity(reduced_mat)
        # Calculate the clustering coefficient for each word
        transitivities = functions.get_clustering_coefficient(
            cos_sim, verbose, percentile)
        # get polysemy as 1 - transitivity
        polysemy_score = [1 - transitivity for transitivity in transitivities]
        # Create a DataFrame for the clustering coefficients
        current_polysemy_score = pd.DataFrame(
            data=polysemy_score, index=reduced_vocab,
            columns=[f"polysemy_score_{year}"])
        # add clustering coefficients if already exists
        if 'polysemy_score_years' in locals():
            polysemy_score_years = pd.merge(
                left=polysemy_score_years, right=current_polysemy_score,
                left_index=True, right_index=True, how='inner')
        # create as new dataframe if doesn't exist yet
        else:
            polysemy_score_years = current_polysemy_score.copy()
    # save the resulting dataframe to a pickle
    polysemy_score_years.to_pickle(output_file)


def get_polysemy_score_evolution_eng(
    start_year = 1960, end_year = 2000,
    folder = OUTPUT_FOLDER, input_folder = INPUT_FOLDER,
    concreteness_filename = CONCRETENESS_PICKLE_ENG, verbose = False,
    percentile=90):
    """Create dataframe of polysemy score for historical english word
    embeddings.

    Parameters:
    start_year (int): The starting year (inclusive)
    end_year (int): The ending year (inclusive)
    folder (str): The folder where the output file will be saved
    input_folder (str): The folder of the word embeddings
    concreteness_filename (str): The name of the concreteness file for
    reducing the vocabulary
    verbose (bool): If True, print status messages
    percentile (int): only connections above percentile will be used.
    """
    # define the output file path
    output_file = os.path.join(folder, 'polysemy_score_years_eng.pkl')
    # read the concreteness file to dataframe
    concreteness = pd.read_pickle(os.path.join(folder, concreteness_filename))
    # iterate decades
    for year in range(start_year, end_year+10, 10):
        if verbose:
            print(f'current year: {year}')
        # read embeddings of year
        file = os.path.join(input_folder, 'historical_american_english',
                            f'{year}.txt')
        embeddings = pd.read_csv(file, skiprows=1, sep=' ', header=None,
                                 index_col=0)
        embeddings.index = embeddings.index.str.split('_').str[0]
        # filter for words in concreteness
        embeddings = embeddings[embeddings.index.isin(concreteness.index)]
        # remove duplicates
        embeddings = embeddings[~embeddings.index.duplicated(keep='first')]
        # get polysemy score from cosine similarity
        cos_sim = cosine_similarity(embeddings)
        transitivities = functions.get_clustering_coefficient(
            cos_sim, verbose, percentile)
        polysemy_score = [1 - transitivity for transitivity in transitivities]
        # create dataframe from polysemy score
        current_polysemy_score = pd.DataFrame(
            data=polysemy_score, index=embeddings.index,
            columns=[f'polysemy_score_{year}'])
        # add current result to dataframe
        if 'polysemy_score_years_eng' in locals():
            polysemy_score_years_eng = pd.merge(
                left=polysemy_score_years_eng, right=current_polysemy_score,
                left_index=True, right_index=True, how='inner')
        else:
        # create new dataframe if it doesn't exist yet
            polysemy_score_years_eng = current_polysemy_score.copy()
    # save result to pickle
    polysemy_score_years_eng.to_pickle(output_file)


def process_eng_concreteness(
        folder = CONCRETENESS_FOLDER_ENG, filename = CONCRETENESS_FILE_ENG,
        output = OUTPUT_FOLDER, output_filename = CONCRETENESS_PICKLE_ENG):
    """Read English concreteness file, filter and save to pickle.

    Parameters:
    folder (str): folder of the concreteness file
    filename (str): of the concreteness file
    output (str): folder for the output
    output_filename (str): name of the output pickle
    """
    # get the filepaths for the input and output
    output_file = os.path.join(output, output_filename)
    input_file = os.path.join(folder, filename)
    # read the input
    concreteness = pd.read_csv(input_file, sep='\t', index_col='Word')
    # remove words that are not known by all participants
    concreteness = concreteness[concreteness['Percent_known'] == 1]
    # filter type of words
    dom_pos = ['Noun', 'Adjective', 'Verb', 'Adverb']
    concreteness = concreteness[concreteness['Dom_Pos'].isin(dom_pos)]
    # remove bigrams
    concreteness = concreteness[concreteness['Bigram'] == 0]
    # rename column and save to pickle
    concreteness.rename(columns={'Conc.M': 'concreteness'}, inplace=True)
    concreteness[['concreteness']].to_pickle(output_file)


if __name__ == "__main__":
    #process_ger_concreteness()
    #process_non_conformity()
    #get_polysemy_score_evolution(percentile=80, verbose=True)
    #get_polysemy_score_evolution(
    #   language='eng',
    #   percentile=90,
    #   verbose=True,
    #   vocabulary=pd.read_pickle(
    #       os.path.join('data', 'processed', 'concreteness_eng.pkl')).index
    #   )
    #get_polysemy_score_evolution_eng(percentile=90)
    process_eng_concreteness()
