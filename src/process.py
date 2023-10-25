import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config, utils


def get_hist_polysemy_score(
        input_dir,
        input_archive,
        output_dir,
        output_file,
        start_year=1950, end_year=1990,
        percentile=90,
        vocabulary=None,
        force_process=False,
        verbose=False
):

    # define the output directory and file, create if doesn't exist
    # output_dir = os.path.join(processed_data_dir, language)
    output_file = os.path.join(output_dir, output_file)

    if not os.path.exists(output_file) or force_process:
        # create output directory if doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        polysemy_score_years = pd.DataFrame()
        # iterate decades
        for year in range(start_year, end_year+10, 10):
            if verbose:
                print("Year:", year)
            # load the matrix and vocabulary for the current year
            archive = os.path.join(input_dir, input_archive)
            mat = utils.load_mat(year, archive)
            vocab = utils.load_vocab(year, archive)

            if verbose:
                print("Ratio of non-zero vectors:", utils.check_sparcity(mat))
            # remove zero vectors from matrix
            mat, vocab = utils.remove_empty_words(mat, vocab)
            # if vocabulary is passed, remove all other words
            if vocabulary is not None:
                mat, vocab = utils.reduce_to_list_of_words(
                    mat, vocab, vocabulary)
            # calculate the cosine similarity between all word-vectors
            cos_sim = cosine_similarity(mat)
            # calculate the clustering coefficient for each word
            transitivities = utils.get_clustering_coefficient(
                cos_sim, verbose, percentile)
            # get polysemy as 1 - transitivity
            polysemy_score = [
                1 - transitivity for transitivity in transitivities]
            # create a DataFrame for the current clustering coefficients
            current_polysemy_score = pd.DataFrame(
                data=polysemy_score, index=vocab,
                columns=[f"polysemy_score_{year}"])
            if verbose:
                print(
                    f"polysemy score of {year} contains {len(current_polysemy_score)} words")
            # add clustering coefficients
            polysemy_score_years = pd.merge(
                left=polysemy_score_years, right=current_polysemy_score,
                left_index=True, right_index=True, how='outer')

        polysemy_score_years['slope'] = polysemy_score_years.apply(utils.get_slope_of_clustering_coeff, axis=1)
        # save the resulting dataframe to a pickle
        polysemy_score_years.to_csv(output_file, sep=';')

    return polysemy_score_years.index



def get_contemp_polysemy_score(
        input_dir, input_archive, output_dir, output_file,
        percentile=90,
        vocabulary=None,
        force_process=False, verbose=False):

    output_file = os.path.join(output_dir, output_file)

    if not os.path.exists(output_file) or force_process:
        # create output directory if doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # load contemp file
    mat, vocab = utils.load_contemp(
        archive=os.path.join(input_dir, input_archive), 
        max_word_length=29) # to be changed, taken from analysing concreteness
    # remove non-zero
    mat, vocab = utils.remove_empty_words(mat, vocab)
    # filter vocab
    if vocabulary is not None:
        mat, vocab = utils.reduce_to_list_of_words(mat, vocab, vocabulary)
    # cosine sim
    cos_sim = cosine_similarity(mat)
    # clustering coeff
    transitivities = utils.get_clustering_coefficient(
        cos_sim, verbose, percentile)

    polysemy_score = [1 - transitivity for transitivity in transitivities]
    # store as csv
    polysemy_score_df = pd.DataFrame(
        data=polysemy_score, index=vocab,
        columns=["contemp_polysemy_score"])

    polysemy_score_df.to_csv(output_file, sep=';')


def get_concreteness_german(
    input_dir = config.CONCRETENESS_FOLDER, 
    input_file = config.CONCRETENESS_FILE,
    output_dir = config.PROCESSED_DATA_DIR, 
    all_lower_case = True):

    # create the path to the input and output files
    output_file = os.path.join(output_dir, "concreteness_ger.csv")
    concreteness_file = os.path.join(input_dir, input_file)
    # read in the concreteness ratings file
    concreteness = pd.read_csv(concreteness_file, sep='\t', index_col='Word')
    if all_lower_case:
        # change all words to lower case
        # this results in duplicates (eg noun and verb)
        concreteness.index = concreteness.index.str.lower()
        # drop duplicates
        concreteness = concreteness[~concreteness.index.duplicated(keep='first')]
    # rename column to concreteness
    concreteness.rename(columns={'AbstConc': 'concreteness'}, inplace=True)
    # save the processed file as a pickle
    concreteness[['concreteness']].to_csv(output_file, sep=';')

    return concreteness.index


def process_all_files_for_language(language, input_dir,
                                   output_dir, percentile=90, force_process=False, verbose=False):

    if language == 'german':
        vocabulary = get_concreteness_german(
            input_dir = input_dir, 
            input_file = config.CONCRETENESS_FILENAMES[language],
            output_dir = output_dir, 
            all_lower_case = True)
    else:
        vocabulary = None
    
    # historic polysemy scores
    vocabulary = get_hist_polysemy_score(
        input_dir=input_dir,
        input_archive=f"{config.HIST_EMBEDDINGS_NAMES[language]}.zip",
        output_dir=output_dir,
        output_file=f"hist_polysemy_score_{language}.csv",
        percentile=percentile,
        vocabulary=vocabulary,
        force_process=force_process,
        verbose=verbose
    )

    get_contemp_polysemy_score(
        input_dir=input_dir,
        input_archive=f"{config.CONTEMP_EMBEDDINGS_IDS[language]}.zip",
        output_dir=output_dir,
        output_file=f"contemp_polysemy_score_{language}.csv",
        vocabulary=vocabulary,
        percentile=percentile,
        force_process=force_process,
        verbose=verbose
    )
