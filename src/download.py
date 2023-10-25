import os

import wget

from src import config



def download_dataset(folder, filename, description, url, force_download):
    file_path = os.path.join(folder, filename)
    
    # only download of file doesn't exist or if force-download flag
    if not os.path.exists(file_path) or force_download:
        print(f"Downloading {description}")
        
        # create the subfolder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # download file/archive
        wget.download(url=url + filename, out=folder)

    else:
        print(f"{description} already exist")


def download_all_files_for_language(language, folder, force_download=False):
    # historic embeddings
    download_dataset(
        folder=folder,
        filename=f"{config.HIST_EMBEDDINGS_NAMES[language]}.zip",
        description="Historic Embeddings",
        url=config.HIST_EMBEDDINGS_URL,
        force_download=force_download
    )

    # contemporary embeddings
    download_dataset(
        folder=folder,
        filename=f"{config.CONTEMP_EMBEDDINGS_IDS[language]}.zip",
        description="Contemporary Embeddings",
        url=config.CONTEMP_EMBEDDINGS_URL,
        force_download=force_download
    )

    # concreteness ratings
    download_dataset(
        folder=folder,
        filename=config.CONCRETENESS_FILENAMES[language],
        description="Concreteness Ratings",
        url=config.CONCRETENESS_URLS[language],
        force_download=force_download
    )
