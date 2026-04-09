import cv2 # OpenCV 라이브러리 임포트
import mediapipe as mp # MediaPipe 라이브러리 임포트

# 1. FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh # FaceMesh 솔루션 가져오기
face_mesh = mp_face_mesh.FaceMesh( # FaceMesh 객체 초기화
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 캡처 설정
cap = cv2.VideoCapture(0)

while cap.isOpened(): # 웹캠이 열려 있는 동안 반복
    ret, frame = cap.read() # 프레임 읽기
    if not ret: break

    height, width, _ = frame.shape # 프레임의 높이와 너비 가져오기
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV는 BGR 형식을 사용하므로 RGB로 변환

    # 3. 랜드마크 추출
    results = face_mesh.process(rgb_frame)

    # 4. 결과 시각화
    if results.multi_face_landmarks: # 얼굴 랜드마크가 검출된 경우
        for face_landmarks in results.multi_face_landmarks: 
            for landmark in face_landmarks.landmark: 
                x = int(landmark.x * width) # 랜드마크의 x 좌표 계산
                y = int(landmark.y * height) # 랜드마크의 y 좌표 계산
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # 랜드마크 위치에 작은 원 그리기

    cv2.imshow('Mediapipe Face Landmark', frame) # 결과 프레임 보여주기
    # 5. ESC(27) 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release() # 웹캠 캡처 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기