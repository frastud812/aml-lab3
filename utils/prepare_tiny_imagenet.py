# Download Tiny-ImageNet into data/ and adjust the val split to be used with ImageFolder.
# Usage (from the repo root): python utils/prepare_tiny_imagenet.py
import os
import shutil
import urllib.request
import zipfile

URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATA_DIR = 'data'
ROOT = os.path.join(DATA_DIR, 'tiny-imagenet-200')


def download_and_extract():
    zip_path = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
    if os.path.isdir(ROOT):
        print(f'{ROOT} already exists, skipping download')
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Downloading {URL} ...')
    urllib.request.urlretrieve(URL, zip_path)

    print('Extracting ...')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
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
