import cv2 # OpenCV 라이브러리 임포트
import numpy as np # NumPy 라이브러리 임포트
from sort import Sort # SORT 추적기 임포트 (

# 1. 모델 및 추적기 초기화
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames() # YOLOv3의 레이어 이름 가져오기
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # 출력 레이어 인덱스 가져오기
tracker = Sort() # SORT 추적기 초기화

# 2. 비디오 캡처 설정
cap = cv2.VideoCapture("slow_traffic_small.mp4")
vehicle_class_ids = [2, 3, 5, 7] # 자동차, 오토바이, 버스, 트럭

while cap.isOpened(): # 비디오가 열려 있는 동안 반복
    ret, frame = cap.read() # 프레임 읽기
    if not ret: break
    height, width, _ = frame.shape # 프레임의 높이와 너비 가져오기

    # 3. 객체 검출 (YOLOv3)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 이미지 전처리
    net.setInput(blob) # 네트워크에 입력 설정
    outs = net.forward(output_layers) # 네트워크 실행하여 출력 가져오기

    boxes, confidences = [], [] # 검출된 객체의 바운딩 박스와 신뢰도 저장할 리스트 초기화
    for out in outs: # 출력 레이어마다 반복
        for detection in out: 
            scores = detection[5:] # 클래스별 신뢰도 점수 가져오기
            class_id = np.argmax(scores) # 가장 높은 신뢰도 점수의 클래스 ID 가져오기
            confidence = scores[class_id] # 해당 클래스의 신뢰도 점수 가져오기
            if confidence > 0.5 and class_id in vehicle_class_ids: # 신뢰도가 0.5 이상이고 차량 클래스인 경우
                center_x, center_y = int(detection[0] * width), int(detection[1] * height) # 바운딩 박스 중심 좌표 계산
                w, h = int(detection[2] * width), int(detection[3] * height) # 바운딩 박스 너비와 높이 계산
                boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h]) # 바운딩 박스 좌표 저장
                confidences.append(float(confidence)) # 신뢰도 저장

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 비최대 억제 적용하여 중복된 박스 제거

    # 4. 객체 추적 (SORT)
    dets = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i] # 바운딩 박스 좌표 가져오기
            dets.append([x, y, x + w, y + h, confidences[i]]) # SORT에 입력할 형식으로 변환하여 저장
    dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5)) # 검출된 객체가 없는 경우 빈 배열 생성
    trackers = tracker.update(dets) # SORT 추적기 업데이트하여 트랙킹된 객체의 바운딩 박스와 ID 가져오기

    # 5. 결과 시각화
    for d in trackers:
        x1, y1, x2, y2, track_id = [int(i) for i in d] # 트랙킹된 객체의 바운딩 박스 좌표와 ID 가져오기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 바운딩 박스 그리기
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # 트랙킹된 객체의 ID 표시

    cv2.imshow("Traffic Object Tracking", frame) # 결과 프레임 보여주기
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release() # 비디오 캡처 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기