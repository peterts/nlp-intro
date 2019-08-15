import psutil
import urllib
from pathlib import Path
import subprocess
import sys
from distutils.spawn import find_executable


WIKI_EXTRACTOR_URL = "https://github.com/attardi/wikiextractor"

WIKI_DUMP_FOLDER = Path("data/wiki")
DUMP_FROM_URL = "https://dumps.wikimedia.org/nowiki/latest/nowiki-latest-pages-articles-multistream.xml.bz2"
DUMP_TO_FILEPATH = WIKI_DUMP_FOLDER / DUMP_FROM_URL.rsplit("/")[-1]
DUMP_TO_EXTRACTED_DIR = WIKI_DUMP_FOLDER / "extracted"

N_WORKERS = psutil.cpu_count(logical=False)


def download_data(rerun=False):
    if not DUMP_TO_FILEPATH.exists() or rerun:
        print(f"Data file '{DUMP_TO_FILEPATH}' not found. Downloading data.")
        if not DUMP_TO_FILEPATH.parent.exists():
            DUMP_TO_FILEPATH.parent.mkdir(parents=True)
        urllib.request.urlretrieve(DUMP_FROM_URL, DUMP_TO_FILEPATH)
    else:
        print(f"Data file '{DUMP_TO_FILEPATH}' already downloaded.")


def extract_data(retry=True, rerun=False):
    if not any(list_all_extracted_files()) or rerun:
        try:
            if not DUMP_TO_EXTRACTED_DIR.exists():
                DUMP_TO_EXTRACTED_DIR.mkdir(parents=True)
            out = str(DUMP_TO_EXTRACTED_DIR.resolve())
            inp = str(DUMP_TO_FILEPATH.resolve())
            script_path = find_executable("WikiExtractor.py")
            subprocess.run([sys.executable, script_path, "-o", out,  "--json", inp], shell=True)
        except FileNotFoundError:
            if retry:
                print("WikiExtractor not found. Installing.")
                subprocess.run(["pip", "install", f"git+{WIKI_EXTRACTOR_URL}"])
                extract_data(False)
    else:
        print("Data has already been extracted.")


def list_all_extracted_files():
    yield from DUMP_TO_EXTRACTED_DIR.rglob("wiki_*")


if __name__ == '__main__':
    download_data()
    extract_data()
