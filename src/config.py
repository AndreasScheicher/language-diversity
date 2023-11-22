import os

DATA_DIR = "data"
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FIGURES_DIR = "figures"

# define directories
#HIST_EMBEDDINGS_DIR = os.path.join(EXTERNAL_DATA_DIR, "hist_embeddings")
#AFFECTIVE_NORMS_GER_DIR = os.path.join(EXTERNAL_DATA_DIR, "affective_norms")
#CORPUS_HISTORICAL_AM_ENG = os.path.join(EXTERNAL_DATA_DIR, "historical_american_english")
#CONCRETENESS_RATINGS_ENG = os.path.join(EXTERNAL_DATA_DIR, "concreteness_ratings_en")
#POLYSEMY_HIST_DIR = os.path.join(PROCESSED_DATA_DIR, "hist_polysemy_score")

# define urls and filenames for download
#MILLION_POST_CORPUS_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/"
#MILLION_POST_CORPUS_FILE = "million_post_corpus.tar.bz2"
#AFFECTIVE_NORMS_GER_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
#AFFECTIVE_NORMS_GER_FILE = "affective_norms.txt.gz"
#CORPUS_HISTORICAL_AM_ENG_URL = "http://vectors.nlpl.eu/repository/20/"
#CORPUS_HISTORICAL_AM_ENG_FILE = "188.zip"
#CONCRETENESS_ENG_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"

LANGUAGES_SUPPORTED = ["english", "french", "german"]

HIST_EMBEDDINGS_URL = "http://snap.stanford.edu/historical_embeddings/"

HIST_EMBEDDINGS_NAMES = {
    "english": "eng-all_sgns",
    "french": "fre-all_sgns",
    "german": "ger-all_sgns"
}

CONTEMP_EMBEDDINGS_URL = "http://vectors.nlpl.eu/repository/20/"

CONTEMP_EMBEDDINGS_IDS = {
    "english": "40",
    "french": "43",
    "german": "45"
}

# todo: change german concretess to Wartena-Paper
CONCRETENESS_FILENAMES = {
    "english": "Concreteness_ratings_Brysbaert_et_al_BRM.txt",
    "french": "13428_2018_1014_MOESM3_ESM.xlsx",
    "german": "MergedConcreteness.csv" #"affective_norms.txt.gz"
}

CONCRETENESS_URLS = {
    "english": "http://crr.ugent.be/papers/",
    "french": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1014-y/MediaObjects/",
    "german": "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/"
}

FREQUENCY_FILENAMES = {
    "english": "Subtlex US_.rar", # "Subtlex%20US.rar",
    "french": "Lexique383.zip",
    "german": "SUBTLEX-DE_txt_cleaned_with_Google00.zip"
}

FREQUENCY_URLS = {
    "english": "https://web.archive.org/web/20160323233749/http://subtlexus.lexique.org/corpus/",
    "french": "http://www.lexique.org/databases/Lexique383/",
    "german": "http://crr.ugent.be/subtlex-de/"
}


#HIST_EMBEDDINGS_SUBDIR = "sgns"


# archived
#CONCRETENESS_FOLDER = os.path.join(EXTERNAL_DATA_DIR, "affective_norms")
#CONCRETENESS_FILE = "affective_norms.txt"
#CONCRETENESS_FOLDER_ENG = os.path.join(EXTERNAL_DATA_DIR, "concreteness_ratings_eng")
#CONCRETENESS_PICKLE_ENG = "concreteness_eng.pkl"
#CONCRETENESS_FILE_ENG = "Concreteness_ratings_Brysbaert_et_al_BRM.txt"
#MILLION_POSTS_FOLDER = os.path.join(EXTERNAL_DATA_DIR, "million_post_corpus")
#CORPUSDB = "corpus.sqlite3"
