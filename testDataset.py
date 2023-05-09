import cv2
import os

image_name='1.jpg'
image_path=os.path.join('D:\\kfashion\\train\\images',image_name)
label_path=os.path.join('D:\\kfashion\\train\\labels',image_name.replace('.jpg','.txt'))

label_dict={'0':'top','1':'bottom','2':'outer','3':'onepiece'}

image=cv2.imread(image_path)
h,w,_=image.shape
with open(label_path,'r') as f:
    labels=f.readlines()
for label in labels:
    values=label.split(' ')
    x1=int((float(values[1])-(float(values[3])/2))*w)
    y1=int((float(values[2])-(float(values[4])/2))*h)
    x2=int((float(values[1])+(float(values[3])/2))*w)
    y2=int((float(values[2])+(float(values[4])/2))*h)
    image=cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0), 2)
    image=cv2.putText(image,label_dict[values[0]],(x1,y1+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

cv2.imshow('test',image)
cv2.waitKey(0)