# Download Tiny-ImageNet into data/ and adjust the val split to be used with ImageFolder.
# Usage (from the repo root): python utils/prepare_tiny_imagenet.py
import http.client
import os
import shutil
import urllib.request
import zipfile

URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATA_DIR = 'data'
ROOT = os.path.join(DATA_DIR, 'tiny-imagenet-200')


def download(url, path, retries=10):
    # The server often drops the connection mid-download:
    # on each retry, ask only for the missing bytes with an HTTP Range request
    for attempt in range(1, retries + 1):
        # a partial zip has no central directory at the end, so this is False until complete
        if zipfile.is_zipfile(path):
            return

        done = os.path.getsize(path) if os.path.exists(path) else 0
        if done > 0:
            print(f'Attempt {attempt}/{retries}: resuming from {done / 1e6:.0f} MB')
        request = urllib.request.Request(url, headers={'Range': f'bytes={done}-'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # 206: the server resumes from `done`; 200: it ignored Range and restarts from 0
                mode = 'ab' if response.status == 206 else 'wb'
                with open(path, mode) as f:
                    shutil.copyfileobj(response, f, 1024 * 1024)
        except (OSError, http.client.HTTPException) as e:
            print(f'Attempt {attempt}/{retries} interrupted: {e}')

    if not zipfile.is_zipfile(path):
        raise RuntimeError(f'Download failed after {retries} attempts: run the script again to resume')


def download_and_extract():
    zip_path = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
    if os.path.isdir(ROOT):
        print(f'{ROOT} already exists, skipping download')
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Downloading {URL} ...')
    download(URL, zip_path)

    # Extract into a temporary folder and rename at the end,
    # so an interrupted extraction never leaves a half-filled ROOT behind
    print('Extracting ...')
    tmp_dir = os.path.join(DATA_DIR, '_extracting')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    os.rename(os.path.join(tmp_dir, 'tiny-imagenet-200'), ROOT)
    os.rmdir(tmp_dir)
    os.remove(zip_path)


def reorganize_val():
    # The val images are all in val/images, with labels in val_annotations.txt:
    # move each image into val/{class}/ like the train split
    val_dir = os.path.join(ROOT, 'val')
    images_dir = os.path.join(val_dir, 'images')
    if not os.path.isdir(images_dir):
        print('val split already reorganized, skipping')
        return

    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.move(os.path.join(images_dir, fn), os.path.join(val_dir, cls, fn))

    shutil.rmtree(images_dir)


if __name__ == '__main__':
    download_and_extract()
    reorganize_val()
    print('Done')
