import cv2
import requests
import numpy as np
import time
# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냅니다.

# 서버 URL 설정
# server_url = 'http://127.0.0.1:5000/video_feed'  # 서버의 Flask 엔드포인트 URL
server_url = "https://f927-160-238-37-212.ngrok-free.app/video_feed"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode('.jpg', frame)

    # POST 요청을 사용하여 프레임을 서버로 전송
    response = requests.post(server_url, data=img_encoded.tobytes(), headers={'Content-Type': 'image/jpeg'})

    # 디텍딩 이미지 표시를 위한 코드
    # frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    # cv2.imshow("image", frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    print(response.content)
    
    