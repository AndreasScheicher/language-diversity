import os
import wget
import zipfile
import tarfile
import gzip


DATA_FOLDER = os.path.join("data", "external")
EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, "embeddings")
AFFECTIVE_NORMS_GER_FOLDER = os.path.join(DATA_FOLDER, "affective_norms")

EMBEDDINGS_URL = "http://snap.stanford.edu/historical_embeddings/"
MILLION_POST_CORPUS_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/"
MILLION_POST_CORPUS_FILE = "million_post_corpus.tar.bz2"
AFFECTIVE_NORMS_GER_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
AFFECTIVE_NORMS_GER_FILE = "affective_norms.txt.gz"


def download_embeddings(EMBEDDINGS_FOLDER=EMBEDDINGS_FOLDER, EMBEDDINGS_URL=EMBEDDINGS_URL, languages = ['DE']):
    
    # name of zip
    embeddings_list = {
        "EN": "eng-all_sgns",
        "FR": "fre-all_sgns",
        "DE": "ger-all_sgns"
    }

    for language in languages:
        embeddings_sub_folder = os.path.join(EMBEDDINGS_FOLDER, embeddings_list[language])

        # create subfolder
        if not os.path.exists(embeddings_sub_folder):
            os.makedirs(embeddings_sub_folder)
        
        embedding_zip = embeddings_list[language] + ".zip"

        wget.download(EMBEDDINGS_URL + embedding_zip)
        
        # unzip
        with zipfile.ZipFile(embedding_zip, 'r') as zip_ref:
            zip_ref.extractall(embeddings_sub_folder)
        
        os.remove(embedding_zip)


def download_million_post_corpus(folder=DATA_FOLDER, url=MILLION_POST_CORPUS_URL, file=MILLION_POST_CORPUS_FILE):
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    wget.download(url + file)

    with tarfile.open(file, "r:bz2") as tar:
        tar.extractall(folder)

    os.remove(file)


def download_affective_norms_ger(folder=AFFECTIVE_NORMS_GER_FOLDER, url=AFFECTIVE_NORMS_GER_URL, file=AFFECTIVE_NORMS_GER_FILE):
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    wget.download(url + file)

    output_file = os.path.join(folder, file.rstrip('.gz'))

    with gzip.open(file, 'rb') as f:
        with open(output_file, 'wb') as out:
            out.write(f.read())

    os.remove(file)



if __name__ == "__main__":
    download_million_post_corpus()
    download_embeddings()
    download_affective_norms_ger()
