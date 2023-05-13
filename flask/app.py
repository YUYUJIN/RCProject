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
import cv2
import numpy as np
import time

# Const
SAVE_DIR=increment_path(Path(ROOT / 'storage'),exist_ok=True)
IMAGE_SIZE=640
IMAGE_STRIDE=32
AUTO=True
CONF_THOLD=0.3
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
labels_dict={0:'top',1:'bottom',2:'outer',3:'onepiece'}
API_key=os.environ.get('WEATHER_API_KEY')
WEATHER_URL=f'http://api.weatherapi.com/v1/current.json?key={API_key}&q=Seoul&aqi=yes'
load_dotenv()

# Model load
weights_path='.\\weights\\last.pt'

#device=select_device(device)
model=attempt_load(weights_path,device=DEVICE)
model.conf=CONF_THOLD

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

@app.route('/upload', methods=['POST'])
def test():
    if request.method=='POST':
        target=request.get_json()
        img=np.array(list(target['data'].values()), np.uint8).reshape((target['height'],target['width'],target['bpp']))
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        im,img=preprocess(img,DEVICE)
        
        pred=model(im,augment=False)
        pred=non_max_suppression(pred,conf_thres=model.conf)
        det=pred[0].to(DEVICE)
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for i,(*xyxy, conf, cls) in enumerate(reversed(det)):
                filename=time.strftime("%Y%m%d-%H%M%S-")+str(i)+'.jpg'
                clss = int(cls)
                save_one_box(xyxy, img, file=SAVE_DIR / filename, BGR=False)

        response={
                'message': 'success'
        }       

    return jsonify(response),200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)