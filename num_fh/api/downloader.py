import urllib.request
import os
from os import path
from tqdm import tqdm


NFH_URL = "https://storage.googleapis.com/ai2i/datasets/num_fh/"
NFH_DIR = '.nfh'
IDENTIFICATION_NFH = 'best.pkl'
RESOLUTION_NFH = 'model.tar.gz'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_models():
    home = path.expanduser("~")
    models_path = path.join(home, NFH_DIR)

    if not path.exists(models_path):
        print('couldn\'t fine nfh directory... creating one [directory]')
        os.makedirs(models_path, exist_ok=True)

    identification_path = path.join(home, NFH_DIR, IDENTIFICATION_NFH)
    if not path.exists(identification_path):
        print('couldn\'t find the identification model. downloading one [identification model]')
        download_url(NFH_URL + IDENTIFICATION_NFH, identification_path)

    resolution_path = path.join(home, NFH_DIR, RESOLUTION_NFH)
    if not path.exists(resolution_path):
        print('couldn\'t find the resolution model. downloading one [resolution model]')
        download_url(NFH_URL + RESOLUTION_NFH, resolution_path)


# download_models()
