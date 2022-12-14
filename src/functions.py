import os
import pickle
import numpy as np
import pandas as pd
import igraph as ig
from bs4 import BeautifulSoup
from scipy.stats import linregress
#import preprocess_datasets

EMBEDDINGS_FOLDER = os.path.join("data", "external", "embeddings", "ger-all_sgns")

def load_mat(year=1990, path=EMBEDDINGS_FOLDER):
    file = os.path.join(path, str(year) + "-w.npy") 
    return np.load(file, mmap_mode="c")


def load_vocab(year=1990, path=EMBEDDINGS_FOLDER):
    file = os.path.join(path, str(year) + "-vocab.pkl")
    with open(file, 'rb') as f:
        return pickle.load(f)



def check_sparcity(m):
    return sum(m.any(axis=1)) / m.shape[0]


def remove_empty_words(mat, vocab):
    filled_columns = mat.any(axis=1)
    reduced_mat = np.delete(mat, ~filled_columns, axis=0)
    reduced_vocab = np.array(vocab)[filled_columns]
    return reduced_mat, reduced_vocab


def get_clustering_coefficient(cos_sim, reduced_mat, verbose=False):
    # process for graph
    cutoff_value = np.percentile(cos_sim - np.diag(np.diag(cos_sim)), 90)
    if verbose: print("Cutoff value:", cutoff_value)
    # triangle matrix for removing duplicate cosine similiarities
    triangle = np.tri(cos_sim.shape[0], cos_sim.shape[1], -1)
    # set duplicates and below cutoff value to zero
    above_thresh = np.where(cos_sim >= cutoff_value, cos_sim * triangle, np.zeros(cos_sim.shape))
    # get indices of non-zero values
    indices = np.nonzero(above_thresh)
    # indices should contain ~5% of the cosine similarity matrix
    if verbose: print("Ratio elements above cutoff value", indices[0].shape[0] / ( cos_sim.shape[0] * cos_sim.shape[1] ))

    # create graph
    n_vertices = reduced_mat.shape[0]
    edges = [(a, b) for a, b in zip(indices[0], indices[1])]
    g = ig.Graph(n_vertices, edges, directed=False)
    # get clutering coeff
    transitivities = g.transitivity_local_undirected()
    
    return transitivities




def get_vocab_from_embeddings(start_year=1950, end_year=1990):
    all_vocab = set()
    for year in range(start_year, end_year+10, 10):
        v = set(load_vocab())
        all_vocab = all_vocab.union(v)
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
    
    slope, intercept, r_value, p_value, std_err = linregress(years, coeffs)
    return slope