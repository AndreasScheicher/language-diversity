"""Contains all functions to evaluate the processed data sets."""

import os
import pickle

import numpy as np
import pandas as pd
import igraph as ig
from scipy.stats import linregress

# define folders and default values
EMBEDDINGS_FOLDER = os.path.join("data", "external", "embeddings")
DEFAULT_LANGUAGE_FOLDER = "ger-all_sgns"
DEFAULT_YEAR = 1990


def load_mat(year=DEFAULT_YEAR, path=EMBEDDINGS_FOLDER,
             language_folder=DEFAULT_LANGUAGE_FOLDER):
    """Load the embeddings matrix"""
    file = os.path.join(path, language_folder, f"{year}-w.npy")
    return np.load(file, mmap_mode="c")


def load_vocab(year=DEFAULT_YEAR, path=EMBEDDINGS_FOLDER,
               language_folder=DEFAULT_LANGUAGE_FOLDER):
    """Load the embeddings vocabulary"""
    filepath = os.path.join(path, language_folder, f"{year}-vocab.pkl")
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def check_sparcity(matrix):
    """Get the number of non-zero vectors in matric"""
    return sum(matrix.any(axis=1)) / matrix.shape[0]


def remove_empty_words(mat, vocab):
    """
    Remove empty words from a matrix and a corresponding vocabulary.

    Parameters:
    mat: array of the word embeddings
    vocab: array containing the words in the vocabulary

    Returns:
    reduced_mat: word embeddings of the non-zero vectors
    reduced_vocab: corresponding vocabulary
    """
    # find columns that contain at least one non-zero element
    filled_columns = mat.any(axis=1)
    # delete the columns that do not contain any non-zero element
    reduced_mat = np.delete(mat, ~filled_columns, axis=0)
    # get words in the vocabulary with at least one non-zero element
    reduced_vocab = np.array(vocab)[filled_columns]
    return reduced_mat, reduced_vocab


def reduce_to_list_of_words(mat, vocab, list_of_words):
    """Reduce word embeddings and vocabulary to specified list

    Parameters:
    mat: word embeddings matrix
    vocab: word embeddings vocabulary
    list_of_words: list of words to keep

    Returns:
    reduced_mat: word embeddings in both lists
    reduced_vocab: corresponding vocabulary
    """
    # get words in both lists
    keep_words = [word in list_of_words for word in vocab]
    # delete all other words
    reduced_mat = np.delete(mat, ~np.array(keep_words), axis=0)
    reduced_vocab = vocab[keep_words]
    return reduced_mat, reduced_vocab


def get_clustering_coefficient(cos_sim, verbose=False, percentile=90):
    """Create graph from the cosine similarities above the percentile
    and return the clustering coefficient.

    Parameters:
    cos_sim: cosine similarity matrix
    verbose: if true, print status messages
    percentile: defines the cutoff value of cosine similarites which
    will be used for the graph

    Returns:
    transitivities: of each word
    """
    # find cos_sim cutoff value, depending on percentile
    np.fill_diagonal(cos_sim, 0.)
    cutoff_value = np.percentile(cos_sim, percentile)
    if verbose:
        print("Cutoff value:", cutoff_value)
    # create adjacency matric
    above_thresh = np.where(cos_sim >= cutoff_value, 1, 0)
    # get indices of non-zero values
    indices = np.nonzero(above_thresh)
    # check if number of indices match percentile
    if verbose:
        print("Ratio elements above cutoff value",
              indices[0].shape[0] / (cos_sim.shape[0] * cos_sim.shape[1]))
    # create graph from adjacency matrix
    graph = ig.Graph.Adjacency(above_thresh, mode='lower')
    # get clutering coeff
    transitivities = graph.transitivity_local_undirected()
    return transitivities


def get_vocab_from_embeddings(start_year=1950, end_year=1990):
    """Collect all words from the vocabularly lists for range of years.

    Parameters:
    start_year: beginning of the years range (including)
    end_year: end of the years range (including)

    Returns:
    all_vocab: a list of all the words that appeared in any of the
    vocabularly lists for the specified range of years
    """
    # Create an empty set to store the vocabularly
    all_vocab = set()
    # Iterate over the range of years
    for year in range(start_year, end_year+10, 10):
        # Load the vocabularly for the current year
        vocabulary = set(load_vocab(year=year))
        # Add the words in this vocabularly to the set of all words
        all_vocab = all_vocab.union(vocabulary)
    # Convert the set of all words to a list and return it
    all_vocab = list(all_vocab)
    return all_vocab


def get_slope_of_clustering_coeff(clustering_coeff_df):
    """Calculate the slope of the clustering coefficients
    for each year in the given DataFrame.

    Parameters:
    clustering_coeff_df: A DataFrame with the clustering coefficients
    for each year, where the index is the year and the columns are
    the clustering coefficients.

    Returns:
    slope: of the clustering coefficients over time.
    """
    # get years from dataframe column names
    years = [int(name[-4:]) for name in clustering_coeff_df.index]
    coeffs = clustering_coeff_df.values
    # get slope from linear regression model
    slope = linregress(years, coeffs)[0]
    return slope
