import cv2 # OpenCV 라이브러리 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트
import glob # 파일 경로 패턴 매칭을 위한 glob 임포트

# 체크보드 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성 (3D world points)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) # (0,0,0), (1,0,0)...
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # x, y 좌표 생성
objp *= square_size # 실제 크기로 스케일링

# 저장할 좌표 리스트
objpoints = [] # 실제 세계의 3D 점들
imgpoints = [] # 이미지 상의 2D 점들

# 이미지 경로 설정
images = glob.glob("0312/calibration_images/left*.jpg")

img_size = None # 이미지 크기 (width, height) - 첫 번째 이미지에서 추출하여 사용

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images: # 이미지 파일 경로 반복
    img = cv2.imread(fname) # 이미지 로드
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 그레이스케일 변환
    
    if img_size is None: # 첫 번째 이미지에서 크기 추출
        img_size = gray.shape[::-1] # (width, height)

    # 이미지에서 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 검출된 경우에만 데이터 추가
    if ret == True:
        objpoints.append(objp)

        # 코너 좌표 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2) # 이미지 상의 2D 점들 저장

cv2.destroyAllWindows() # 모든 OpenCV 창 닫기

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 카메라 행렬 K와 왜곡 계수 dist 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:") # 카메라 행렬 K 출력
print(K)

print("\nDistortion Coefficients:") # 왜곡 계수 dist 출력
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 사용자가 요청한 이미지 로드
target_image = "0312/calibration_images/left02.jpg"
img = cv2.imread(target_image)

if img is not None:
    # 이미지 왜곡 보정
    dst = cv2.undistort(img, K, dist, None, K)

    # 원본과 결과 비교를 위해 가로로 이어 붙이기
    result = np.hstack((img, dst))
    
    cv2.imshow('Distortion Correction (Original vs Undistorted)', result) # 결과 시각화
    print(f"\n{target_image} 이미지가 성공적으로 보정되었습니다.") # 성공 메시지 출력
    cv2.waitKey(0) # 키 입력 대기
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기
else:
    print(f"이미지를 찾을 수 없습니다: {target_image}") # 이미지 로드 실패 메시지 출력