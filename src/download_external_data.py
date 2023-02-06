import os
import wget
import zipfile
import tarfile
import gzip


DATA_FOLDER = os.path.join("data", "external")
EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, "embeddings")
AFFECTIVE_NORMS_GER_FOLDER = os.path.join(DATA_FOLDER, "affective_norms")
CORPUS_HISTORICAL_AM_EN = os.path.join(DATA_FOLDER, "historical_american_english")
CONCRETENESS_RATINGS_EN = os.path.join(DATA_FOLDER, "concreteness_ratings_en")

EMBEDDINGS_URL = "http://snap.stanford.edu/historical_embeddings/"
MILLION_POST_CORPUS_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/"
MILLION_POST_CORPUS_FILE = "million_post_corpus.tar.bz2"
AFFECTIVE_NORMS_GER_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
AFFECTIVE_NORMS_GER_FILE = "affective_norms.txt.gz"
CORPUS_HISTORICAL_AM_EN_URL = "http://vectors.nlpl.eu/repository/20/"
CORPUS_HISTORICAL_AM_EN_FILE = "188.zip"
CONCRETENESS_RATINGS_EN_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"


def download_embeddings(EMBEDDINGS_FOLDER=EMBEDDINGS_FOLDER, EMBEDDINGS_URL=EMBEDDINGS_URL, languages = 'DE'):
    """
    Downloads the specified language's pre-trained word embeddings from EMBEDDINGS_URL and saves them to 
    the specified EMBEDDINGS_FOLDER.

    Parameters:
        EMBEDDINGS_FOLDER (str): The path to the folder where the downloaded word embeddings will be saved.
        EMBEDDINGS_URL (str): The URL where the word embeddings are hosted.
        languages (str or list): A string or list of strings indicating which language's word embeddings to download.
    """

    # Define a dictionary mapping language codes to folder names
    embeddings_list = {
        "EN": "eng-all_sgns",
        "FR": "fre-all_sgns",
        "DE": "ger-all_sgns"
    }

    # Convert the input language codes to a list if it is a string
    if isinstance(languages, str):
        languages = [languages]

    # Iterate over the list of languages
    for language in languages:
        # Create a subfolder for the language if it doesn't already exist
        embeddings_sub_folder = os.path.join(EMBEDDINGS_FOLDER, embeddings_list[language])
        if not os.path.exists(embeddings_sub_folder):
            os.makedirs(embeddings_sub_folder)
        
        # Download the zip file for the language
        embedding_zip = embeddings_list[language] + ".zip"
        wget.download(EMBEDDINGS_URL + embedding_zip)
        
        # Extract the files from the zip file to the subfolder
        with zipfile.ZipFile(embedding_zip, 'r') as zip_ref:
            zip_ref.extractall(embeddings_sub_folder)
        
        # Delete the zip file
        os.remove(embedding_zip)



def download_million_post_corpus(folder=DATA_FOLDER, url=MILLION_POST_CORPUS_URL, file=MILLION_POST_CORPUS_FILE):
    """
    Downloads the Million Post Corpus and extracts it to the specified folder.

    Args:
        folder (str): The directory where the corpus will be extracted.
        url (str): The URL where the corpus can be downloaded from.
        file (str): The name of the corpus file.
    """

    # Create the target folder if it doesn't already exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the corpus file
    wget.download(url + file)

    # Extract the corpus from the downloaded tar file
    with tarfile.open(file, "r:bz2") as tar:
        tar.extractall(folder)

    # Delete the tar file after extraction is complete
    os.remove(file)


def download_affective_norms_ger(folder=AFFECTIVE_NORMS_GER_FOLDER, url=AFFECTIVE_NORMS_GER_URL, file=AFFECTIVE_NORMS_GER_FILE):
    """Downloads a file from a specified URL, unzips it, and saves it to a specified folder on the local filesystem.

    Args:
        folder: The local directory where the downloaded file should be saved.
        url: The URL from which the file should be downloaded.
        file: The file name.

    Returns:
        None
    """
    # Check if the specified folder exists and create it if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the file using the wget module
    wget.download(url + file, file)

    # Specify the output file name
    output_file = os.path.join(folder, file.rstrip('.gz'))

    # Open the downloaded file with the gzip module
    with gzip.open(file, 'rb') as f:
        # Open the output file in write binary mode
        with open(output_file, 'wb') as out:
            # Write the unzipped file to the output file
            out.write(f.read())

    # Remove the original zipped file
    os.remove(file)


def download_corpus_historical_american_english(folder=CORPUS_HISTORICAL_AM_EN, url=CORPUS_HISTORICAL_AM_EN_URL, file=CORPUS_HISTORICAL_AM_EN_FILE):
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


def download_concreteness_eng(folder=CONCRETENESS_RATINGS_EN, url=CONCRETENESS_RATINGS_EN_URL):
    # Check if the specified folder exists and create it if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download the file using the wget module
    wget.download(url, out=folder)


if __name__ == "__main__":
    #download_million_post_corpus()
    #download_embeddings()
    #download_affective_norms_ger()
    #download_corpus_historical_american_english()
    #download_concreteness_eng()
    pass