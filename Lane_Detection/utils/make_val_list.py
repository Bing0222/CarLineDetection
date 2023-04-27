from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from config import ConfigVal 
from os.path import join
import pandas as pd
import os
from sklearn.utils import shuffle
from tqdm import tqdm

"""
main
"""
if __name__ == '__main__':
    cfg = ConfigVal()

    image_list = []
    label_list = []
    image_root = join(cfg.IMAGE_ROOT, '')
    label_root = join(cfg.LABEL_ROOT, '')
    for d1 in tqdm(os.listdir(cfg.IMAGE_ROOT)):
        # Road02, ...
        d1_image_root = join(cfg.IMAGE_ROOT, d1)
        if not os.path.isdir(d1_image_root):
            continue
        d1_label_root = join(cfg.LABEL_ROOT, 'Label_' + d1.lower(), 'Label')

        for d2 in os.listdir(d1_image_root):
            # Record001, ...
            d2_image_root = join(d1_image_root, d2)
            if not os.path.isdir(d2_image_root):
                continue
            d2_label_root = join(d1_label_root, d2)

            for d3 in os.listdir(d2_image_root):
                # 'Camera 5', ...
                d3_image_root = join(d2_image_root, d3)
                if not os.path.isdir(d3_image_root):
                    continue
                d3_label_root = join(d2_label_root, d3)

                for file in os.listdir(d3_image_root):
                    if not file.endswith('.jpg'):
                        continue
                    label_file_name = file.replace('.jpg', '_bin.png')
                    if not os.path.exists(join(d3_label_root, label_file_name)):
                        continue
                    imagefile = join(d3_image_root, file).replace(image_root, '')
                    labelfile = join(d3_label_root, label_file_name).replace(label_root, '')

                    image_list.append(imagefile)
                    label_list.append(labelfile)

    print(len(image_list), len(label_list))
    df = pd.DataFrame({'image':image_list, 'label':label_list})
    df = shuffle(df)
    df = df[0:cfg.VAL_MAX_NUM]
    df.to_csv(join(cfg.DATA_LIST_ROOT, 'fs_val.csv'), index=False)
