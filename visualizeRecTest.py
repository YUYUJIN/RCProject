import cv2
import json
import glob
import os

category=['상의','하의','아우터','원피스']
category_dict={'상의':'top','하의':'bottom','아우터':'outer','원피스':'one piece'}

image_path='D:\\New_sample\\image\\li\\11894.jpg'

image_name=image_path.split('\\')[-1].split('.')[0]
json_paths=glob.glob(os.path.join('D:\\New_sample\\label','*',image_name+'.json'))
with open(json_paths[0],'r',encoding='utf-8') as j:
    json_data=json.load(j)
rec_data=json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']
    
image=cv2.imread(image_path)

for c in category:
    for rec in rec_data[c]:
        if rec=={}:
            continue
        else:
            x=int(rec['X좌표'])
            y=int(rec['Y좌표'])
            w=int(rec['가로'])
            h=int(rec['세로'])
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 2)
            image=cv2.putText(image,category_dict[c],(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

cv2.imshow('test',image)
cv2.waitKey(0)