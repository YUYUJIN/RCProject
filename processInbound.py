import os
import glob
from tqdm import tqdm

labels_root='D:\\kfashion\\valid\\labels'
labels_paths=glob.glob(os.path.join(labels_root,'*'))
for labels_path in tqdm(labels_paths):
    with open(labels_path,'r') as f:
        labels=f.readlines()
    os.remove(labels_path)
    for label in labels:
        values=label.split(' ')
        w=float(values[3])
        h=float(values[4])
        if w>1.0:
            w=1.0
        if h>1.0:
            h=1.0
        with open(labels_path,'a',encoding='utf-8') as f:
            f.write(f'{values[0]} {values[1]} {values[2]} {w} {h} \n')