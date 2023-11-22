"""Contains all functions to evaluate the processed data sets."""

import os
import io
import pickle
import zipfile

import numpy as np
# import pandas as pd
import igraph as ig
from scipy.stats import linregress
from scipy.sparse import csr_matrix

# define folders and default values
# EMBEDDINGS_FOLDER = os.path.join("data", "external", "embeddings")
# DEFAULT_LANGUAGE_FOLDER = "ger-all_sgns"
# DEFAULT_YEAR = 1990


def load_mat(year, archive):
    """Load the embeddings matrix"""
    # file = os.path.join(path, f"{year}-w.npy")
    # return np.load(file, mmap_mode="c")

    with zipfile.ZipFile(archive, 'r') as z:
        with z.open(f'sgns/{year}-w.npy') as f:
            buf = io.BytesIO(f.read())
            return np.load(buf)  # , mmap_mode="c")


def load_vocab(year, archive):
    """Load the embeddings vocabulary"""
    # filepath = os.path.join(path, f"{year}-vocab.pkl")
    # with open(filepath, 'rb') as file:
    #    return pickle.load(file)

    with zipfile.ZipFile(archive, 'r') as z:
        with z.open(f'sgns/{year}-vocab.pkl') as f:
            buf = io.BytesIO(f.read())
            return pickle.load(buf)


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
    mat = np.delete(mat, ~filled_columns, axis=0)
    # get words in the vocabulary with at least one non-zero element
    vocab = np.array(vocab)[filled_columns]
    return mat, vocab


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
    mat = np.delete(mat, ~np.array(keep_words), axis=0)
    vocab = vocab[keep_words]
    return mat, vocab


def get_clustering_coefficient(adjacency, verbose=False, percentile=90):
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
    np.fill_diagonal(adjacency, 0.)
    cutoff_value = np.percentile(adjacency, percentile)
    if verbose:
        print("Cutoff value:", cutoff_value)
    # create adjacency matrix
    adjacency = np.where(adjacency >= cutoff_value, 1, 0)
    # get indices of non-zero values
    # indices = np.nonzero(adjacency)
    # check if number of indices match percentile
    # if verbose:
    #    print("Ratio elements above cutoff value",
    #          indices[0].shape[0] / (cos_sim.shape[0] * cos_sim.shape[1]))
    # remove duplicate values by only considering lower triangle
    adjacency = np.tril(adjacency, -1)
    # use sparse representation to lower memory requirement
    adjacency = csr_matrix(adjacency)
    # create graph from adjacency matrix
    graph = ig.Graph.Adjacency(adjacency.toarray().tolist())
    # get clutering coeff
    transitivities = graph.transitivity_local_undirected()
    return transitivities


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


def load_contemp(archive, max_word_length=100):

    # first pass: determine dimensions
    num_rows = 0
    num_cols = 0

    with zipfile.ZipFile(archive, "r") as z:
        stream = z.open("model.txt")
        for idx, line in enumerate(stream):
            if idx == 0:  # skip first line
                continue
            parts = line.decode('utf-8', errors='ignore').strip().split()
            word = parts[0]
            if idx == 1:  # determine the number of columns from second line
                num_cols = len(parts) - 1
            if len(word) > max_word_length:  # skip words longer than max_word_length
                continue
            # count lines
            if num_cols == len(parts) - 1:
                num_rows += 1

    # initialize empty arrays
    vocab = np.empty(num_rows, dtype=f'<U{max_word_length}')
    mat = np.empty((num_rows, num_cols), dtype=np.float32)

    # second pass: fill arrays
    current_row = 0
    with zipfile.ZipFile(archive, "r") as z:
        stream = z.open("model.txt")
        for idx, line in enumerate(stream):
            if idx == 0:  # skip first line
                continue
            parts = line.decode('utf-8', errors='ignore').strip().split()
            word = parts[0]
            if len(word) > max_word_length:
                continue
            if num_cols != len(parts) - 1:
                continue
            vocab[current_row] = word
            mat[current_row] = np.array(parts[1:], dtype=np.float32)
            current_row += 1

    vocab, mat

    return mat, vocab
