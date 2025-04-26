import os
import tarfile
import numpy as np
import webdataset as wds
import tqdm
from PIL import Image
from pathlib import Path
import torch.utils.data as data
import requests

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as file, tqdm.tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(32768):
            file.write(data)
            bar.update(len(data))

class CLEVRTEX(data.Dataset):
    BASE_URL = 'https://thor.robots.ox.ac.uk/datasets/clevrtex/'
    FILES = [
        'clevrtex_full_part1.tar.gz',
        'clevrtex_full_part2.tar.gz',
        'clevrtex_full_part3.tar.gz',
        'clevrtex_full_part4.tar.gz',
        'clevrtex_full_part5.tar.gz',
        # 'clevrtex_outd.tar.gz',
        # 'clevrtex_camo.tar.gz'
    ]

    def __init__(self, root, split='train', download=True):
        self.root = root
        self.split = split
        self.image_dir = os.path.join(self.root, 'clevrtex_full')

        if download and not self._check_existing_data():
            self._download()

        self.images, self.semsegs = self._load_data()
    
    def _check_existing_data(self):
        clevrtex_full_exists = os.path.exists(self.image_dir)
        return clevrtex_full_exists

    def _download(self):
        mkdir_if_missing(self.root)

        for file in self.FILES:
            file_path = os.path.join(self.root, file)
            if not os.path.isfile(file_path):
                print(f'Downloading {file}...')
                download_file(self.BASE_URL + file, file_path)

            print(f'Extracting {file_path}...')
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
            print(f'Extracted {file_path}.')
            os.remove(file_path) 

    def _load_data(self):
        all_images = []
        all_semsegs = []

        folder_indices = list(range(50))
        val_indices = folder_indices[:10]
        train_indices = folder_indices[10:]

        indices = train_indices if self.split == 'train' else val_indices

        for i in indices:
            folder_path = os.path.join(self.image_dir, str(i))
            for img_file in Path(folder_path).glob('CLEVRTEX_full_*.png'):
                if img_file.name.endswith('.png') and '_flat' not in img_file.name:
                    semseg_file = img_file.with_name(img_file.name.replace('.png', '_flat.png'))

                    # Проверяем, существует ли соответствующая маска
                    if semseg_file.is_file():
                        all_images.append(img_file)
                        all_semsegs.append(semseg_file)
                    # else:
                    #     continue
                    #     print(f"Warning: Segmentation file does not exist for {img_file}: {semseg_file}")

        print(f'Loaded {len(all_images)} images and {len(all_semsegs)} segmentation masks for {self.split}')
        return all_images, all_semsegs

    def get_samples(self, load_annotations=True, convert_images_to_numpy=True):
        for img_path, semseg_path in zip(self.images, self.semsegs):
            key = os.path.basename(img_path).split('.')[0]
            
            if convert_images_to_numpy:
                img_key = "image.npy"
                with Image.open(img_path).convert('RGB') as img:
                    img_data = np.asarray(img, dtype="uint8")
            else:
                img_key = "image" + os.path.splitext(img_path)[-1]
                with open(img_path, "rb") as f:
                    img_data = f.read()

            sample = {"__key__": key, img_key: img_data}
            
            if load_annotations:
                if convert_images_to_numpy:
                    semseg = np.array(Image.open(semseg_path), dtype=np.uint8)
                    sample["segmentations.npy"] = semseg
                else:
                    with open(semseg_path, "rb") as f:
                        semseg_data = f.read()
                    sample["segmentations.png"] = semseg_data
            
            yield sample

def write_dataset(image_dir, out_dir, split='train', load_annotations=True, convert_images_to_numpy=True, maxcount=256):
    mkdir_if_missing(out_dir)
    clevrtex = CLEVRTEX(root=image_dir, split=split)

    pattern = os.path.join(out_dir, f"clevrtex-{split}-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for sample in tqdm.tqdm(clevrtex.get_samples(load_annotations, convert_images_to_numpy), 
                                 total=len(clevrtex.images), 
                                 desc=f"Creating {split} shards"):
            sink.write(sample)
    
    print(f"Finished writing shards to {out_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sharded dataset from CLEVRTEX.")
    parser.add_argument("--download-dir", default="./data", help="Directory to download CLEVRTEX data")
    parser.add_argument("--out-path", default="./shards", help="Directory where shards are written")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Dataset split to use")
    parser.add_argument("--maxcount", type=int, default=1000,
                        help="Maximum samples per shard")
    parser.add_argument("--original-image-format",
                        action="store_true",
                        help="Whether to keep the original image encoding (e.g. jpeg), or convert to numpy")
    parser.add_argument("--load-anno", action="store_true", default=True,
                        help="Load annotation data (segmentation masks)")

    args = parser.parse_args()

    write_dataset(
        image_dir=args.download_dir,
        out_dir=args.out_path,
        split=args.split,
        load_annotations=args.load_anno,
        convert_images_to_numpy=not args.original_image_format,
        maxcount=args.maxcount
    )
