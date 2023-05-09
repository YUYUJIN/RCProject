import os
import glob
import shutil
import json
from tqdm import tqdm

labels_dict={'상의':0,'하의':1,'아우터':2,'원피스':3}

image_root='D:\\kfashion\\K-Fashion\\Validation\\원천데이터'
label_root='D:\\kfashion\\K-Fashion\\Validation\\라벨링데이터'
image_save_path='D:\\kfashion\\valid\\images'
label_save_path='D:\\kfashion\\valid\\labels'

image_paths=glob.glob(os.path.join(image_root,'*','*.jpg'))
for image_path in tqdm(image_paths):
    # file name meta
    temp=image_path.split('\\')
    folder=temp[-2]
    file_name=temp[-1]
    
    # move image file
    shutil.move(image_path,os.path.join(image_save_path,file_name))

    # label json read and processing
    json_path=os.path.join(label_root,folder,file_name.replace('.jpg','.json'))
    with open(json_path,'r',encoding='utf-8') as j:
        json_data=json.load(j)
    meta=json_data['이미지 정보']
    w=meta['이미지 너비']
    h=meta['이미지 높이']
    rec_data=json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']
    
    # save label data in yolo format
    for key in rec_data:
        for rec in rec_data[key]:
            if rec=={}:
                continue
            
            # yolo format
            yolo_w=round(rec['가로']/w,5)
            yolo_h=round(rec['세로']/h,5)
            yolo_x=round((rec['X좌표']+(rec['가로']/2))/w,5)
            yolo_y=round((rec['Y좌표']+(rec['세로']/2))/h,5)

            txt_path=os.path.join(label_save_path,file_name.replace('.jpg','.txt'))
            with open(txt_path,'a',encoding='utf-8') as f:
                f.write(f'{labels_dict[key]} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')