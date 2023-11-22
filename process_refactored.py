import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Other imports and configurations as in your original script

def get_most_frequent_words(input_dir, input_file, language, nr_words=20_000):
    file_path = os.path.join(input_dir, input_file)

    if language == 'english':
        concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        most_frequent_words = concreteness.nlargest(nr_words, 'SUBTLEX', keep="all").index
    elif language == 'german':
        concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        most_frequent_words = concreteness.nlargest(nr_words, 'WFfreqcount', keep="all").index
    elif language == 'french':
        concreteness = pd.read_csv(file_path, sep='\t', index_col=0)
        most_frequent_words = concreteness.nlargest(nr_words, 'freqlivres', keep="all").index
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
    else:
        concreteness = pd.read_csv(concreteness_file, sep='\t', index_col='Word')

    if all_lower_case:
        concreteness.index = concreteness.index.str.lower()
        concreteness = concreteness[~concreteness.index.duplicated(keep='first')]

    # Rename columns and save
    concreteness.rename(columns={'AbstConc': 'concreteness'}, inplace=True)
    concreteness[['concreteness']].to_csv(output_file, sep=';')
    return concreteness.index

# Similar functions for get_hist_polysemy_score, get_contemp_polysemy_score

def process_all_files_for_language(language, input_dir, output_dir, percentile=90, force_process=False, verbose=False):
    if language == 'english':
        vocabulary = get_most_frequent_words(input_dir, config.CONCRETENESS_FILENAMES['english'], 'english', 20_000)
        get_concreteness('english', input_dir, config.CONCRETENESS_FILENAMES['english'], output_dir, True)
    elif language == 'french':
        vocabulary = get_most_frequent_words(input_dir, config.CONCRETENESS_FILENAMES['french'], 'french', 20_000)
        get_concreteness('french', input_dir, config.CONCRETENESS_FILENAMES['french'], output_dir, True)
    elif language == 'german':
        vocabulary = get_most_frequent_words(input_dir, config.CONCRETENESS_FILENAMES['german'], 'german', 20_000)
        get_concreteness('german', input_dir, config.CONCRETENESS_FILENAMES['german'], output_dir, True)
    else:
        vocabulary = None

    return vocabulary


def calculate_polysemy_scores(input_dir, input_archive, output_dir, output_file, vocabulary, percentile, verbose, historical=False, start_year=1950, end_year=1990):
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_file)

    if not os.path.exists(output_file_path) or force_process:
        polysemy_score_years = pd.DataFrame()

        if historical:
            year_range = range(start_year, end_year + 10, 10)
        else:
            year_range = [None]  # No specific year for contemporary data

        for year in year_range:
            if verbose:
                print("Year:", year)
            archive_path = os.path.join(input_dir, input_archive)
            mat, vocab = utils.load_data(archive_path, year)

            if verbose:
                print("Ratio of non-zero vectors:", utils.check_sparcity(mat))
            mat, vocab = utils.remove_empty_words(mat, vocab)

            if vocabulary is not None:
                mat, vocab = utils.reduce_to_list_of_words(mat, vocab, vocabulary)

            cos_sim = cosine_similarity(mat)
            transitivities = utils.get_clustering_coefficient(cos_sim, verbose, percentile)
            polysemy_score = [1 - transitivity for transitivity in transitivities]

            col_name = f"polysemy_score_{year}" if historical else "contemp_polysemy_score"
            current_polysemy_score = pd.DataFrame(data=polysemy_score, index=vocab, columns=[col_name])

            if verbose:
                print(f"Polysemy score of {year} contains {len(current_polysemy_score)} words")

            polysemy_score_years = pd.merge(left=polysemy_score_years, right=current_polysemy_score, left_index=True, right_index=True, how='outer')

        if historical:
            polysemy_score_years['slope'] = polysemy_score_years.apply(utils.get_slope_of_clustering_coeff, axis=1)

        polysemy_score_years.to_csv(output_file_path, sep=';')

def process_all_files_for_language(language, input_dir, output_dir, percentile=90, force_process=False, verbose=False):
    # Assuming the rest of the function implementation is similar to the previous part

    # historic polysemy scores
    calculate_polysemy_scores(
        input_dir=input_dir,
        input_archive=f"{config.HIST_EMBEDDINGS_NAMES[language]}.zip",
        output_dir=output_dir,
        output_file=f"hist_polysemy_score_{language}.csv",
        vocabulary=vocabulary,
        percentile=percentile,
        verbose=verbose,
        historical=True
    )

    # contemporary polysemy scores
    calculate_polysemy_scores(
        input_dir=input_dir,
        input_archive=f"{config.CONTEMP_EMBEDDINGS_IDS[language]}.zip",
        output_dir=output_dir,
        output_file=f"contemp_polysemy_score_{language}.csv",
        vocabulary=vocabulary,
        percentile=percentile,
        verbose=verbose,
        historical=False
    )

# Include any other utility functions and configurations as required

# Rest of your code...
