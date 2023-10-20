"""Download all data sets necessary for the project."""

import os

import zipfile
import wget

from src import config

def download_hist_embeddings(languages = 'english',
        folder=config.EMBEDDINGS_DIR, url=config.EMBEDDINGS_URL,
        embeddings_names=config.EMBEDDINGS_NAMES):
    """Download the pre-trained word embeddings.

    Parameters:
    folder (str): The folder where the word embeddings will be saved.
    url (str): The URL where the word embeddings are hosted.
    languages (str or list): The language(s) which will be downloaded.
    """

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

def download_contemp_embeddings():
    # get embeddings from here: http://vectors.nlpl.eu/repository/
    pass

def download_concreteness_ratins():
    # get concreteness ratings from here: https://github.com/billdthompson/cogsci-auto-norm
    pass
