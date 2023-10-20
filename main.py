import os
import argparse

from src import config, download #, preprocess, functions #, analysis, visualize


def ensure_directories_exist(languages):
    """Ensure required directories and language-specific subdirectories are present."""

    # Base directories to check
    dirs_to_check = [
        config.EXTERNAL_DATA_DIR,
        config.PROCESSED_DATA_DIR
    ]

    for dir_path in dirs_to_check:
        #for language in languages:
        #    lang_subdir = os.path.join(dir_path, language)

        #    if not os.path.exists(lang_subdir):
        #        os.makedirs(lang_subdir)
        #        print(f"Created directory: {lang_subdir}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)




def main(languages, force_download=False, force_process=False):
    """Main function to run the entire process."""

    # Ensure necessary folders exist
    ensure_directories_exist(languages)


    for language in languages:

        # download embeddings if don't exist
        embeddings_lang_dir = os.path.join(config.EMBEDDINGS_DIR, language)

        if not os.path.exists(embeddings_lang_dir) or force_download:
            print(f"Downloading data for {language}...")
            download.download_hist_embeddings(language)

        """
        if not os.path.exists(os.path.join(config.PROCESSED_DATA_DIR, f"{language}_processed.ext")) or force_process:
            print(f"Preprocessing data for {language}...")
            preprocess_datasets.preprocess(language)

        # Perform analysis on preprocessed data
        print(f"Running analysis for {language}...")
        results, graphs = analysis.run(language)
        
        # Visualize the results
        print(f"Visualizing results for {language}...")
        visualize.display(results, graphs)
        """



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on language datasets.")
    parser.add_argument("-l", "--languages", help="Specify the languages to analyze", choices=config.LANGUAGES_SUPPORTED, nargs='+', default=['english'])
    parser.add_argument("--force-download", help="Force re-download of datasets", action="store_true")
    parser.add_argument("--force-process", help="Force re-processing of datasets", action="store_true")
    args = parser.parse_args()

    main(languages=args.languages, force_download=args.force_download, force_process=args.force_process)

