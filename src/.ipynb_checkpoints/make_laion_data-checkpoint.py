import os
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
import io
from urllib.request import urlopen

DIR_LAION_META = Path('./../laion_meta_file')
DIR_OUTPUT = Path('./../data/laion')
RESIZE_SIZE = 256
SAVE_IMG_NUM_PER_DIR = 1000000
SEED = 0


def read_img_from_url(url: str):
    flg = False
    try:
        file =io.BytesIO(urlopen(url).read())
        img = Image.open(file)
        flg = True
    except:
        img = None
    return flg, img


def make_random_idx_list(df_imgs: pd.DataFrame):
    img_num = len(df_imgs)
    list_index = list(range(img_num))
    random.shuffle(list_index)
    return list_index

if __name__ == '__main__':
    random.seed(SEED)
    path_laion_meta_files = list(DIR_LAION_META.glob('*.parquet'))
    split_num = 0
    for path_laion_meta_file in tqdm(path_laion_meta_files):
        
        output_path = DIR_OUTPUT / f'split{split_num}'
        os.makedirs(output_path, exist_ok=True)
        
        print(f'read {path_laion_meta_file}...')
        df_imgs = pd.read_parquet(path_laion_meta_file)
        
        list_index = make_random_idx_list(df_imgs)
        save_num = 0
        skip_num = 0
        print("")
        print(f'start split{split_num}/{len(path_laion_meta_files)-1}')
        for index in list_index:
            is_read, img = read_img_from_url(df_imgs.iloc[index]['URL'])
            if is_read == False:
                skip_num += 1
                print(f'skip num: {skip_num}', end="\r")
                continue

            img = img.resize((RESIZE_SIZE, RESIZE_SIZE))
            img = img.convert('RGB')
            img.save(output_path / f'{save_num}.png')

            caption = df_imgs.iloc[index]['TEXT']
            with open(output_path / f'{save_num}.txt', 'w') as f:
                f.write(caption)

            save_num += 1
            if save_num == SAVE_IMG_NUM_PER_DIR:
                break
        print(f'end split{split_num}/{len(path_laion_meta_files)-1}')
        print("")
        split_num += 1