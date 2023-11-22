import os
import argparse

import pandas as pd

from src import config, download, process #, functions #, analysis, visualize


def main(languages, force_download=False, force_process=False, verbose=False):

    # iterate chosen languages
    for language in languages:

        # define subdir of language
        data_ext_lang_subdir = os.path.join(config.EXTERNAL_DATA_DIR, language)
        processed_subdir = os.path.join(config.PROCESSED_DATA_DIR, language)

        if not os.path.exists(data_ext_lang_subdir) or force_download:
            print(f"Downloading data for {language}...")

            download.download_all_files_for_language(
                language=language, folder=data_ext_lang_subdir, 
                force_download=force_download)


        # get polysemy cutoff value
        if language == 'english':
            vocabulary = process.get_most_frequent_words(
                input_dir=data_ext_lang_subdir, 
                input_file=config.FREQUENCY_FILENAMES[language], 
                language=language, 
                nr_words=10_000
            )

            cutoff_percentiles = [96, 98]
            for cutoff_percentile in cutoff_percentiles:
                process.get_contemp_polysemy_score(
                    input_dir=data_ext_lang_subdir,
                    input_archive=f"{config.CONTEMP_EMBEDDINGS_IDS[language]}.zip",
                    output_dir=processed_subdir,
                    output_file=f"contemp_polysemy_score_{language}_{cutoff_percentile}.csv",
                    vocabulary=vocabulary,
                    percentile=cutoff_percentile,
                    force_process=force_process,
                    verbose=verbose
                )

            #contemp = pd.read_csv(os.path.join(processed_subdir, f'contemp_polysemy_score_{language}.csv'), sep=';', index_col=0)
            #polysemy_reference = pd.read_csv(os.path.join(processed_subdir, "concreteness_w_definition.csv"), usecols=["Word"])
            #polysemy_reference = polysemy_reference.value_counts().reset_index()
            #polysemy_reference.set_index('Word', drop=True, inplace=True)
            #merged = contemp.merge(polysemy_reference, how='inner', left_index=True, right_index=True)
            #print(f"cutoff percentile: {cutoff_percentile}")
            #print(merged.corr('spearman'))

        break
        # get polysemy score from embeddings if don't exist
        processed_subdir = os.path.join(config.PROCESSED_DATA_DIR, language)
        if not os.path.exists(processed_subdir) or force_process:
            print(f"Calculating historic polysemy scores for {language}...")
            
            process.process_all_files_for_language(
                language=language, input_dir=data_ext_lang_subdir,
                output_dir=processed_subdir, force_process=force_process,
                verbose=verbose)

        """
        # Perform analysis on preprocessed data
        print(f"Running analysis for {language}...")
        results, graphs = analysis.run(language)
        
        # Visualize the results
        print(f"Visualizing results for {language}...")
        visualize.display(results, graphs)
        """



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on language datasets.")
    parser.add_argument("-l", "--languages", help="Specify the languages to analyze", choices=config.LANGUAGES_SUPPORTED, nargs='+', default=['german'])
    parser.add_argument("--force-download", help="Force re-download of datasets", action="store_true")
    parser.add_argument("--force-process", help="Force re-processing of datasets", action="store_true")
    parser.add_argument("--verbose", help="Print outputs", action="store_true")
    args = parser.parse_args()

    main(languages=args.languages, force_download=args.force_download, force_process=args.force_process, verbose=args.verbose)
