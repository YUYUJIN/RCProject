import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import save_one_box
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression,scale_boxes,increment_path

from flask import Flask, render_template, Response,jsonify,request,send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from db import Database
import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

# Const
SAVE_DIR=increment_path(Path(ROOT / 'image'),exist_ok=True)
SAVE_PATH='/image'
IMAGE_SIZE=640
IMAGE_STRIDE=32
AUTO=True
CONF_THOLD=0.3
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
labels_dict={0:'top',1:'bottom',2:'outer',3:'onepiece'}
weather_dict_key=['temp_c','is_day','wind_kph','pressure_mb','humidity','cloud','feelslike_c','precip_mm']
table_columns=['CATEGORY','IMAGE','TEMP','DAY','WIND','PRESS','HUMI','CLOUD','FEELS','PRECI']
MAXIUM_VALUE={'temp_c':40,'is_day':1,'wind_kph':20,'pressure_mb':1200,'humidity':100,'cloud':100,'feelslike_c':40,'precip_mm':200}
RECOMMEND_THOLD=0.8

# For weather API
import requests
from ast import literal_eval
load_dotenv()
API_key=os.environ.get('WEATHER_API_KEY')
WEATHER_URL=f'http://api.weatherapi.com/v1/current.json?key={API_key}&q=Seoul&aqi=yes'

# Model load
weights_path='.\\weights\\last.pt'

app = Flask(__name__)
CORS(app)

# For test
@app.route('/')
def index():
    return render_template('index.html')

def preprocess(image,device):
    im=letterbox(image,IMAGE_SIZE,stride=IMAGE_STRIDE,auto=AUTO)[0] # padded resize
    im=im.transpose((2,0,1))[::-1] # BGR to RGB
    im=np.ascontiguousarray(im) # contiguous
    im=torch.from_numpy(im).to(device)
    im=im.float()
    im/=255
    if len(im.shape)==3:
        im=im[None]
    return im,image

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory('./image', filename)

@app.route('/recommend',methods=['POST'])
def recommend():
    if request.method=='POST':
        target=request.get_json()
        
        # weather api
        params={'lat':target['lat'],'lon':target['lon']}
        weather=requests.get(WEATHER_URL,params=params)
        weather_data=literal_eval(weather.content.decode('utf-8'))['current']
        current_state=[weather_data[key]/MAXIUM_VALUE[key] for key in weather_dict_key]
        
        clothes_data=db.getData(table_name=os.environ.get('TABLENAME'))
        
        coordinates=[]
        for i,cd in enumerate(clothes_data):
            cos_sim=cosine_similarity(np.array([current_state]),np.array([cd[3:]]))
            if cos_sim>RECOMMEND_THOLD:
                coordinates.append([i,cos_sim])
        coordinates.sort(key=lambda x:x[1])
        if len(coordinates)<10:
            result=[clothes_data[cd[0]][2] for cd in coordinates]
        else:
            result=[clothes_data[cd[0]][2] for cd in coordinates[:10]]

        response={
            'images': result,
            'message': 'success'
        } 
    
    return jsonify(response),200

@app.route('/upload', methods=['POST'])
def upload():
    if request.method=='POST':
        target=request.get_json()

        # weather api
        params={'lat':target['lat'],'lon':target['lon']}
        weather=requests.get(WEATHER_URL,params=params)
        weather_data=literal_eval(weather.content.decode('utf-8'))['current']

        # image
        img=np.array(list(target['data'].values()), np.uint8).reshape((target['height'],target['width'],target['bpp']))
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        im,img=preprocess(img,DEVICE)
        
        # model run
        pred=model(im,augment=False)
        pred=non_max_suppression(pred,conf_thres=model.conf)
        det=pred[0].to(DEVICE)
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for i,(*xyxy, conf, cls) in enumerate(reversed(det)):
                filename=time.strftime("%Y%m%d-%H%M%S-")+str(i)+'.jpg'
                save_one_box(xyxy, img, file=SAVE_DIR / filename, BGR=False)

                file_path=SAVE_PATH+'/'+filename
                clss = labels_dict[int(cls)]

                insert_data=[clss,file_path]
                insert_data+=[weather_data[key]/MAXIUM_VALUE[key] for key in weather_dict_key]
                db.insData(table_name=os.environ.get('TABLENAME'),columns=table_columns,values=insert_data)

        response={
                'message': 'success'
        }       

    return jsonify(response),200

@app.route('/upload/beggin',methods=['POST'])
def uploadB():
    if request.method=='POST':
        target=request.get_json()

        # weather api
        weather_data=[float(target[key]) for key in weather_dict_key]

        # image
        img=np.array(list(target['data'].values()), np.uint8).reshape((target['height'],target['width'],target['bpp']))
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        im,img=preprocess(img,DEVICE)
        
        # model run
        pred=model(im,augment=False)
        pred=non_max_suppression(pred,conf_thres=model.conf)
        det=pred[0].to(DEVICE)
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for i,(*xyxy, conf, cls) in enumerate(reversed(det)):
                filename=time.strftime("%Y%m%d-%H%M%S-")+str(i)+'.jpg'
                save_one_box(xyxy, img, file=SAVE_DIR / filename, BGR=False)

                file_path=SAVE_PATH+'/'+filename
                clss = labels_dict[int(cls)]

                insert_data=[clss,file_path]
                insert_data+=[weather_data[i]/MAXIUM_VALUE[key] for i,key in enumerate(weather_dict_key)]
                db.insData(table_name=os.environ.get('TABLENAME'),columns=table_columns,values=insert_data)

        response={
                'message': 'success'
        }       

    return jsonify(response),200

if __name__ == '__main__':
    db=Database(host=os.environ.get('HOST'),port=int(os.environ.get('PORT')),user=os.environ.get('USERNAME'),password=os.environ.get('PASSWARD'),database_name=os.environ.get('DATABASENAME'))

    #device=select_device(device)
    model=attempt_load(weights_path,device=DEVICE)
    model.conf=CONF_THOLD

    app.run(host='0.0.0.0',port=5000,debug=True)