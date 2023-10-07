from flask import Flask, request, Response
from pyngrok import conf, ngrok
import cv2
import numpy as np
import json
import time
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

PORT = 8000

app = Flask(__name__)





WEIGHTS = 'yolov7-tiny.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


# Webcam
cap = cv2.VideoCapture(0)

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


# Detect function
def detect(frame):
    # Load image
    img0 = frame

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]

    # s = ''
    # s += '%gx%g ' % img.shape[2:]  # print string


    detects = []
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()  # detections per class
        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # label = f'{names[int(cls)]} {conf:.2f}'
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            detects.append(
                [names[int(cls)], c1, c2]
            )
            # print(c1, c2)
            # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        
        # print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

    # return results
    return float(f"{time.time() - t0:.3f}"), detects

@app.route("/", methods=["GET"])
def main():
    return ""

average = 0
cnt = 0
@app.route('/video_feed', methods=['POST'])
def video_feed():

    image_data = request.data  # 클라이언트에서 POST로 전송한 이미지 데이터
    print("DD")
    # 이미지 데이터를 NumPy 배열로 디코딩
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # frame을 이용하여 객체 탐지 또는 다른 작업 수행
    with torch.no_grad(): 
        time, result = detect(frame)
    
    
    # # 결과 프레임을 클라이언트로 전송 (예시: 클라이언트에게 전송할 이미지 데이터를 반환)
    # print(result)
    # _, img_encoded = cv2.imencode('.jpg', frame)
    # return Response(response=img_encoded.copy().tobytes(), status=200, mimetype='image/jpeg')

    # 출력되 결과만을 클라이언트로 전송
    result = {"data": result}
    return Response(response=json.dumps(result), status=200, mimetype="application/json") 
if __name__ == '__main__':
    # http_tunnel =   ngrok.connect(PORT)
    # tunnels = ngrok.get_tunnels() # ngrok forwording 정보
    # for kk in tunnels: # Forwording 정보 출력
    #     print(kk)

    app.run(host='0.0.0.0', port=PORT, debug=False) 
