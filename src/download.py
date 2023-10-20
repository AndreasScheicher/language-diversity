"""Download all data sets necessary for the project."""

import os

import zipfile
import tarfile
import gzip
import wget

from src import config

# define folders
#DATA_FOLDER = os.path.join("data", "external")
#EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, "embeddings")
#AFFECTIVE_NORMS_GER_FOLDER = os.path.join(DATA_FOLDER, "affective_norms")
#CORPUS_HISTORICAL_AM_ENG = os.path.join(DATA_FOLDER, "historical_american_english")
#CONCRETENESS_RATINGS_ENG = os.path.join(DATA_FOLDER, "concreteness_ratings_en")

# define urls and filenames for download
#EMBEDDINGS_URL = "http://snap.stanford.edu/historical_embeddings/"
#MILLION_POST_CORPUS_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/"
#MILLION_POST_CORPUS_FILE = "million_post_corpus.tar.bz2"
#AFFECTIVE_NORMS_GER_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
#AFFECTIVE_NORMS_GER_FILE = "affective_norms.txt.gz"
#CORPUS_HISTORICAL_AM_ENG_URL = "http://vectors.nlpl.eu/repository/20/"
#CORPUS_HISTORICAL_AM_ENG_FILE = "188.zip"
#CONCRETENESS_ENG_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"


def download_embeddings(languages = 'english',
        folder=config.EMBEDDINGS_DIR, url=config.EMBEDDINGS_URL,
        embeddings_names=config.EMBEDDINGS_NAMES):
    """Download the pre-trained word embeddings.

    Parameters:
    folder (str): The folder where the word embeddings will be saved.
    url (str): The URL where the word embeddings are hosted.
    languages (str or list): The language(s) which will be downloaded.
    """

    # Define a dictionary mapping language codes to folder names
    #embeddings_names = {
    #    "EN": "eng-all_sgns",
    #    "FR": "fre-all_sgns",
    #    "DE": "ger-all_sgns"
    #}
    # Convert the input language codes to a list if it is a string
    if isinstance(languages, str):
        languages = [languages]

    for language in languages:
        # Create a subfolder for the language if it doesn't already exist
        embeddings_lang_dir = os.path.join(folder, language)
        if not os.path.exists(embeddings_lang_dir):
            os.makedirs(embeddings_lang_dir)

        # Download the zip file
        embedding_zip = f"{embeddings_names[language]}.zip"
        wget.download(url + embedding_zip)

        # Extract the files from the zip file to the subfolder
        with zipfile.ZipFile(embedding_zip, 'r') as zip_ref:
            zip_ref.extractall(embeddings_lang_dir)

        # Delete the zip file
        os.remove(embedding_zip)

"""
def download_million_post_corpus(
        folder=DATA_FOLDER, url=MILLION_POST_CORPUS_URL,
        file=MILLION_POST_CORPUS_FILE):
    \"""Downloads the Million Post Corpus from DerStandard.

    Parameters:
    folder (str): The directory where the corpus will be extracted.
    url (str): The URL where the corpus is hosted.
    file (str): The name of the corpus file.
    \"""

    # Create the target folder if it doesn't already exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the corpus file
    wget.download(url + file)

    # Extract files from the tar
    with tarfile.open(file, "r:bz2") as tar:
        tar.extractall(folder)

    # Delete the tar file
    os.remove(file)


def download_affective_norms_ger(
        folder=AFFECTIVE_NORMS_GER_FOLDER, url=AFFECTIVE_NORMS_GER_URL,
        file_path=AFFECTIVE_NORMS_GER_FILE):
    \"""Downloads the affective norms dataset.

    Parameters:
    folder (str): where the file will be saved.
    url (str): The URL where the file is hosted.
    file (str): The file name.
    \"""
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the file to the folder
    wget.download(url + file_path, file_path)

    # Specify the output file name
    output_file = os.path.join(folder, file_path.rstrip('.gz'))

    # Open the downloaded file with the gzip module
    with gzip.open(file_path, 'rb') as file:
        # Open the output file in write binary mode
        with open(output_file, 'wb') as out:
            # Write the unzipped file to the output file
            out.write(file.read())

    # Remove the original zipped file
    os.remove(file_path)


def download_corpus_historical_american_english(
        folder=CORPUS_HISTORICAL_AM_ENG, url=CORPUS_HISTORICAL_AM_ENG_URL,
        file=CORPUS_HISTORICAL_AM_ENG_FILE):
    \"""Download the english word embeddings.

    Parameters:
    folder (str): The folder where the word embeddings will be saved.
    url (str): The URL where the word embeddings are hosted.
    file (str): The language(s) which will be downloaded.
    \"""
    # Check if the specified folder exists and create it if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the file using the wget module
    wget.download(url + file, file)

    # Extract the files from the zip file to the subfolder
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(folder)

    # Delete the zip file
    os.remove(file)


def download_concreteness_eng(folder=CONCRETENESS_RATINGS_ENG,
                              url=CONCRETENESS_ENG_URL):
    \"""Download the English concreteness ratings.

    Parameters:
    folder (str): where the file will be saved.
    url (str): The URL where the file is hosted.
    \"""
    # Check if the specified folder exists and create it if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the file using the wget module
    wget.download(url, out=folder)


if __name__ == "__main__":
    #download_million_post_corpus()
    download_embeddings(languages = 'EN')
    #download_affective_norms_ger()
    #download_corpus_historical_american_english()
    #download_concreteness_eng()

    
"""
