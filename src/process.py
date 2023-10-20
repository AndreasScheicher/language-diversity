"""Preprocess all necessary data sets for the project."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config, utils


def get_polysemy_score_evolution(
        start_year = 1950, end_year = 1990, folder=config.PROCESSED_DATA_DIR,
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
        mat = utils.load_mat(year,
                                 language_folder=embeddings_list[language])
        vocab = utils.load_vocab(year,
                                     language_folder=embeddings_list[language])

        if verbose:
            print("Ratio of non-zero vectors:", utils.check_sparcity(mat))
        # Remove zero vectors from matrix
        reduced_mat, reduced_vocab = utils.remove_empty_words(mat, vocab)
        # if vocabulary is passed, remove all other words
        if vocabulary is not None:
            reduced_mat, reduced_vocab = utils.reduce_to_list_of_words(
                reduced_mat, reduced_vocab, vocabulary)
        # Calculate the cosine similarity between all word-vectors
        cos_sim = cosine_similarity(reduced_mat)
        # Calculate the clustering coefficient for each word
        transitivities = utils.get_clustering_coefficient(
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

