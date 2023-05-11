import os
import shutil
from random import sample
from tqdm import tqdm

def sampling(origin_path,save_path,size):
    print('start')
    origin_labels_path=origin_path.replace('images','labels')
    save_labels_path=save_path.replace('images','labels')
    #print(origin_labels_path)
    files=os.listdir(os.path.join(origin_path,))
    for idx in tqdm(sample(range(0,len(files)),size)):
        file=files[idx]
        label=file.replace('.jpg','.txt')
        if os.path.isfile(os.path.join(origin_labels_path,label)):
            shutil.copy(os.path.join(origin_path,file),os.path.join(save_path,file))
            shutil.copy(os.path.join(origin_labels_path,label),os.path.join(save_labels_path,label))
    print('all done')

origin_path='D:\\kfashion\\validOrigin\\images'
save_path='D:\\kfashion\\valid\\images'
sampling(origin_path,save_path,20000)