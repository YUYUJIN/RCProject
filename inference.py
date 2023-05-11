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
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.general import non_max_suppression,scale_boxes
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadImages
from yolov5.models.experimental import attempt_load
import cv2

IMAGE_SIZE=640
CONF_THOLD=0.3
LABEL_HIDDEN=False
CONF_HIDDEN=False
labels_dict={0:'top',1:'bottom',2:'outer',3:'onepiece'}

image_path='.\\testdata\\test4.jpg'
image=cv2.imread(image_path)

weights_path='.\\yolov5\\runs\\train\\exp10\\weights\\last.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
#device=select_device(device)
model=attempt_load(weights_path,device=device)
model.conf=CONF_THOLD

dataset=LoadImages(image_path,IMAGE_SIZE)
for path, im, im0s, vid_cap, s in dataset:
    # image process
    imBox=im0s.copy()
    im=torch.from_numpy(im).to(device)
    im=im.float()
    im/=255
    if len(im.shape)==3:
        im=im[None]

    pred=model(im,augment=False)
    pred = non_max_suppression(pred, conf_thres=model.conf)

    for i,det in enumerate(pred):
        annotator=Annotator(imBox,line_width=2,example=str(model.names))
        det.to(device)

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                clss = int(cls)
                txt= None if LABEL_HIDDEN else (labels_dict[clss] if CONF_HIDDEN else f'{labels_dict[clss]} {conf:.2f}')
                annotator.box_label(xyxy, txt, color=colors(clss, True))
    
    #imBox=cv2.cvtColor(imBox,cv2.COLOR_BGR2RGB)
    cv2.imshow('test',imBox)
    cv2.waitKey(0)

