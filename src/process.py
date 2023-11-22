import os
import io

import numpy as np
import pandas as pd
import openpyxl
import rarfile
import zipfile
from sklearn.metrics.pairwise import cosine_similarity

from src import config, utils

def get_most_frequent_words(input_dir, input_file, language, nr_words=20_000):
    #filename = os.path.splitext(input_file)[0]
    file_path = os.path.join(input_dir, input_file)

    if language == 'english':
        #with rarfile.RarFile(file_path) as opened_rar:
            #opened_rar.extractall(extract_path)
        #    extracted_file = opened_rar.open("Subtlex US/Subtlex.US.txt")
        #    concreteness = pd.read_csv(io.BytesIO(extracted_file.read()))
        #concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        file_path = os.path.join(input_dir, config.CONCRETENESS_FILENAMES[language])
        concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        most_frequent_words = concreteness.nlargest(nr_words, 'SUBTLEX', keep="first").index
    elif language == 'german':
        with zipfile.ZipFile(file_path, 'r') as opened_zip:
            text_file = "SUBTLEX-DE_cleaned_with_Google00.txt"
            with opened_zip.open(text_file) as extracted_file:
                subtlex = pd.read_csv(extracted_file, sep='\t', index_col=0, encoding='ISO-8859-1')
        #extracted_file = os.path.join(input_dir, "SUBTLEX-DE_txt_cleaned_with_Google00", "SUBTLEX-DE_cleaned_with_Google00.txt")
        #concreteness = pd.read_csv(extracted_file, sep='\t', index_col=0, encoding='ISO-8859-1')
        subtlex.index = subtlex.index.map(lambda x: x.lower())
        most_frequent_words = subtlex.nlargest(nr_words, 'WFfreqcount', keep="first").index
    elif language == 'french':
        #with rarfile.RarFile(file_path) as opened_rar:
        #    extracted_file = opened_rar.open("Lexique383.tsv")
        #    concreteness = pd.read_csv(io.BytesIO(extracted_file.read()))
        #concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        with zipfile.ZipFile(file_path, 'r') as opened_zip:
            with opened_zip.open('Lexique383.tsv') as extracted_file:
                lexique = pd.read_csv(io.BytesIO(extracted_file.read()), sep='\t', index_col=0)
        most_frequent_words = lexique.nlargest(nr_words, 'freqlivres', keep="first").index
    else:
        raise ValueError("Unsupported language")

    return most_frequent_words


def get_concreteness(language, input_dir, input_file, output_dir, all_lower_case=True):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"concreteness_{language}.csv")
    concreteness_file = os.path.join(input_dir, input_file)

    # Different reading logic for French
    if language == 'french':
        concreteness = pd.read_excel(concreteness_file, sheet_name='Norms', header=[0, 1], index_col=0)
        # flatten multiindex columns
        concreteness.columns = concreteness.columns.map(lambda col: '_'.join(col) if isinstance(col, tuple) else col)
        concreteness.index.name = 'Word'
    elif language == 'german':
        concreteness = pd.read_csv(concreteness_file, sep=';', index_col=0, header=None)
        concreteness.index.name = 'Word'
    else:
        concreteness = pd.read_csv(concreteness_file, sep='\t', index_col='Word')

    if all_lower_case:
        concreteness.index = concreteness.index.str.lower()
        concreteness = concreteness[~concreteness.index.duplicated(keep='first')]

    # Rename columns and save
    concreteness_col = {
        "english": "Conc.M",
        "french": "Concreteness_mean",
        "german": 1
    }

    concreteness.rename(columns={concreteness_col[language]: 'concreteness'}, inplace=True)
    concreteness[['concreteness']].to_csv(output_file, sep=';')
    #return concreteness.index

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

    #return polysemy_score_years.index



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
    input_dir, 
    input_file = config.CONCRETENESS_FILENAMES,
    output_dir = config.PROCESSED_DATA_DIR, 
    all_lower_case = True
    ):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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

def get_concreteness_french(
    input_dir, 
    input_file = config.CONCRETENESS_FILENAMES,
    output_dir = config.PROCESSED_DATA_DIR, 
    all_lower_case = True
    ):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # create the path to the input and output files
    output_file = os.path.join(output_dir, "concreteness_fre.csv")
    concreteness_file = os.path.join(input_dir, input_file)
    # read in the concreteness ratings file
    concreteness = pd.read_excel(concreteness_file, sheet_name='Norms', header=[0, 1], index_col=0)
    if all_lower_case:
        # change all words to lower case
        # this results in duplicates (eg noun and verb)
        concreteness.index = concreteness.index.str.lower()
        # drop duplicates
        concreteness = concreteness[~concreteness.index.duplicated(keep='first')]
    # rename column to concreteness
    concreteness.columns = concreteness.columns.to_flat_index()
    concreteness.rename(columns={('Concreteness', 'mean'): 'concreteness'}, inplace=True)
    # save the processed file as a pickle
    concreteness[['concreteness']].to_csv(output_file, sep=';')

    return concreteness.index

def get_concreteness_english(
    input_dir, 
    input_file = config.CONCRETENESS_FILENAMES,
    output_dir = config.PROCESSED_DATA_DIR, 
    all_lower_case = True,
    nr_words = None
    ):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # create the path to the input and output files
    output_file = os.path.join(output_dir, "concreteness_eng.csv")
    concreteness_file = os.path.join(input_dir, input_file)
    # read in the concreteness ratings file
    concreteness = pd.read_csv(concreteness_file, sep='\t', index_col=0)
    if nr_words is not None:
        concreteness = concreteness[concreteness['Dom_Pos'] == nr_words]
    
    if all_lower_case:
        # change all words to lower case
        # this results in duplicates (eg noun and verb)
        concreteness.index = concreteness.index.str.lower()
        # drop duplicates
        concreteness = concreteness[~concreteness.index.duplicated(keep='first')]
    # rename column to concreteness
    concreteness.columns = concreteness.columns.to_flat_index()
    concreteness.rename(columns={'Conc.M': 'concreteness'}, inplace=True)
    # save the processed file as a pickle
    concreteness[['concreteness']].to_csv(output_file, sep=';')
    #return concreteness.index

def get_most_frequent_words_english(
    input_dir, 
    input_file,
    nr_words = 20_000
):
    concreteness_file = os.path.join(input_dir, input_file)
    concreteness = pd.read_csv(concreteness_file, sep='\t', index_col=0)

    most_frequent_words = concreteness.nlargest(nr_words, 'SUBTLEX', keep="all")
    
    return most_frequent_words.index

def process_all_files_for_language(language, input_dir,
                                   output_dir, percentile=90, force_process=False, verbose=False):

    vocabulary = get_most_frequent_words(
        input_dir=input_dir, 
        input_file=config.FREQUENCY_FILENAMES[language], 
        language=language, 
        nr_words=20_000
    )

    #get_concreteness('english', input_dir, config.CONCRETENESS_FILENAMES['english'], output_dir, True)
    get_concreteness(
        language = language, input_dir = input_dir, 
        input_file = config.CONCRETENESS_FILENAMES[language], 
        output_dir = output_dir
    )

    """
    if language == 'english':
        vocabulary = get_most_frequent_words_english(
            input_dir = input_dir, 
            input_file = config.CONCRETENESS_FILENAMES[language],
            nr_words = 20_000
        )


        get_concreteness_english(
            input_dir = input_dir, 
            input_file = config.CONCRETENESS_FILENAMES[language],
            output_dir = output_dir, 
            all_lower_case = True
            )
    elif language == 'french':
        vocabulary = get_concreteness_french(
            input_dir = input_dir, 
            input_file = config.CONCRETENESS_FILENAMES[language],
            output_dir = output_dir, 
            all_lower_case = True
            )
    elif language == 'german':
        vocabulary = get_concreteness_german(
            input_dir = input_dir, 
            input_file = config.CONCRETENESS_FILENAMES[language],
            output_dir = output_dir, 
            all_lower_case = True
            )
    else:
        vocabulary = None
    """
    
    # historic polysemy scores
    get_hist_polysemy_score(
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
