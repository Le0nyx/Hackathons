# scripts/download_images.py

import os
import logging
from icrawler.builtin import BingImageCrawler
from icrawler import ImageDownloader
import requests
from requests.exceptions import RequestException

# 1. Configura il logger per vedere solo warning+  
logging.getLogger('icrawler').setLevel(logging.WARNING)

class ValidatingDownloader(ImageDownloader):
    """
    Override del downloader: prima di scrivere, verifica con HEAD che 
    l'URL sia effettivamente un'immagine e non troppo grande.
    """
    def download(self, task, default_ext, timeout=10, **kwargs):
        url = task['file_url']
        try:
            # HEAD per controllare content-type e lunghezza
            head = requests.head(url, timeout=timeout)
            ctype = head.headers.get('Content-Type', '')
            clen  = int(head.headers.get('Content-Length', 0))
            # Filtro: solo jpeg/png e max 5MB
            if ('image' not in ctype) or (clen > 5_000_000):
                return False
        except RequestException:
            return False

        # Se OK, chiama il downloader originale
        return super().download(task, default_ext, timeout=timeout, **kwargs)

def download_images(query, max_num, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=2,  # pochi thread per stabilità
        storage={'root_dir': output_dir},
        downloader_cls=ValidatingDownloader
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
        max_size=None,
        file_idx_offset=0
    )

if __name__ == "__main__":
    # Scarica immagini di basilico e pomodoro
    download_images("basil plant", 500, "data/all_plants/basil")
    download_images("tomato plant", 500, "data/all_plants/tomato")
    print("Download completato!")
