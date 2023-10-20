import os

DATA_DIR = "data"
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# define folders
EMBEDDINGS_DIR = os.path.join(EXTERNAL_DATA_DIR, "embeddings")
AFFECTIVE_NORMS_GER_DIR = os.path.join(EXTERNAL_DATA_DIR, "affective_norms")
CORPUS_HISTORICAL_AM_ENG = os.path.join(EXTERNAL_DATA_DIR, "historical_american_english")
CONCRETENESS_RATINGS_ENG = os.path.join(EXTERNAL_DATA_DIR, "concreteness_ratings_en")

# define urls and filenames for download
EMBEDDINGS_URL = "http://snap.stanford.edu/historical_embeddings/"
MILLION_POST_CORPUS_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/"
MILLION_POST_CORPUS_FILE = "million_post_corpus.tar.bz2"
AFFECTIVE_NORMS_GER_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
AFFECTIVE_NORMS_GER_FILE = "affective_norms.txt.gz"
CORPUS_HISTORICAL_AM_ENG_URL = "http://vectors.nlpl.eu/repository/20/"
CORPUS_HISTORICAL_AM_ENG_FILE = "188.zip"
CONCRETENESS_ENG_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"

LANGUAGES_SUPPORTED = ["english", "french", "german"]

EMBEDDINGS_NAMES = {
    "english": "eng-all_sgns",
    "french": "fre-all_sgns",
    "german": "ger-all_sgns"
}


# archived
CONCRETENESS_FOLDER = os.path.join(EXTERNAL_DATA_DIR, "affective_norms")
CONCRETENESS_FILE = "affective_norms.txt"
CONCRETENESS_FOLDER_ENG = os.path.join(EXTERNAL_DATA_DIR, "concreteness_ratings_eng")
CONCRETENESS_PICKLE_ENG = "concreteness_eng.pkl"
CONCRETENESS_FILE_ENG = "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
MILLION_POSTS_FOLDER = os.path.join(EXTERNAL_DATA_DIR, "million_post_corpus")
CORPUSDB = "corpus.sqlite3"
