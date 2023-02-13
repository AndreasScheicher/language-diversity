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

# load the embeddings
def load_mat(year=DEFAULT_YEAR, path=EMBEDDINGS_FOLDER, language_folder=DEFAULT_LANGUAGE_FOLDER):
    file = os.path.join(path, language_folder, f"{year}-w.npy")
    return np.load(file, mmap_mode="c")

# load the embeddings vocabulary
def load_vocab(year=DEFAULT_YEAR, path=EMBEDDINGS_FOLDER, language_folder=DEFAULT_LANGUAGE_FOLDER):
    filepath = os.path.join(path, language_folder, f"{year}-vocab.pkl")
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# check the sparcity of the embeddings matrix: number of word vectors that contain non-zero elements
def check_sparcity(matrix):
    return sum(matrix.any(axis=1)) / matrix.shape[0]


def remove_empty_words(mat, vocab):
    """
    Remove empty words from a matrix and a corresponding vocabulary.

    Args:
        mat: A 2D NumPy array of shape (m, n) where m is the number of words in the vocabulary
            and n is the size of the embedding vector.
        vocab: A 1D NumPy array of shape (m,) containing the words in the vocabulary.

    Returns:
        A tuple (reduced_mat, reduced_vocab) where reduced_mat is a 2D NumPy array
        of shape (m', n) with m' < m, containing the words in the vocabulary that are not empty,
        and reduced_vocab is a 1D NumPy array of shape (m',) containing the corresponding words.
    """
    # Find the columns  that contain at least one non-zero element
    filled_columns = mat.any(axis=1)
    # Delete the columns that do not contain any non-zero element
    reduced_mat = np.delete(mat, ~filled_columns, axis=0)
    # Select the words in the vocabulary with at least one non-zero element
    reduced_vocab = np.array(vocab)[filled_columns]
    return reduced_mat, reduced_vocab

def reduce_to_list_of_words(mat, vocab, list_of_words):
    keep_words = [word in list_of_words for word in vocab]
    reduced_mat = np.delete(mat, ~np.array(keep_words), axis=0)
    reduced_vocab = vocab[keep_words]
    return reduced_mat, reduced_vocab


def get_clustering_coefficient(cos_sim, verbose=False, percentile=90):
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
        print(
            "Ratio elements above cutoff value",
            indices[0].shape[0] / ( cos_sim.shape[0] * cos_sim.shape[1] )
        )
    # create graph from adjacency matrix
    graph = ig.Graph.Adjacency(above_thresh, mode='lower')
    # get clutering coeff
    transitivities = graph.transitivity_local_undirected()
    return transitivities




def get_vocab_from_embeddings(start_year=1950, end_year=1990):
    """
    Collects all the words from the vocabularly lists for a range of years.

    This function takes in two optional arguments: `start_year` and `end_year`.
    The default values for these arguments are 1950 and 1990, respectively.
    The function iterates over the range of years from `start_year` to `end_year` in
    increments of 10 and collects all the words from the vocabularly lists for these years.
    It then returns a list of all the words that appeared in any of the vocabularly lists.

    Args:
        start_year: An integer representing the first year to include in the range of years.
        end_year: An integer representing the last year to include in the range of years.

    Returns:
        A list of all the words that appeared in any of the vocabularly lists for
        the specified range of years.
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
    return list(all_vocab)



def get_slope_of_clustering_coeff(clustering_coeff_df: pd.DataFrame) -> float:
    """
    Calculate the slope of the clustering coefficients for each year in the given DataFrame.

    Args:
        clustering_coeff_df: A DataFrame with the clustering coefficients for each year,
            where the index is the year and the columns are the clustering coefficients.

    Returns:
        The slope of the clustering coefficients over time.
    """

    years = [int(name[-4:]) for name in clustering_coeff_df.index]
    coeffs = clustering_coeff_df.values

    slope = linregress(years, coeffs)[0]
    return slope
