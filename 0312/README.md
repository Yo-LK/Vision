# 📂 OpenCV 실습
## 01. 체크보드 기반 카메라 캘리브레이션
[Camera Calibration & Image Undistortion]

### 1. 문제 설명

체크보드 패턴이 촬영된 여러 장의 이미지를 분석하여 카메라의 내부 파라미터(K)와 왜곡 계수(Distortion Coefficients)를 추정합니다.

검출된 파라미터를 바탕으로 렌즈에 의해 왜곡된 이미지를 평면으로 보정(Undistortion)하여 시각화합니다.

### 2. 코드

```Python
import cv2
import numpy as np
import glob

# 체크보드 설정 및 실제 좌표 생성
CHECKERBOARD = (9, 6)
square_size = 25.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints = [], []
images = glob.glob("0312/calibration_images/left*.jpg")
img_size = None

# 1. 코너 검출
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None: img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# 2. 캘리브레이션 수행
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 3. 왜곡 보정 시각화
target_image = "0312/calibration_images/left02.jpg"
img = cv2.imread(target_image)
if img is not None:
    dst = cv2.undistort(img, K, dist, None, K)
    result = np.hstack((img, dst))
    cv2.imshow('Calibration Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```
### 3. 해결 방법

코너 검출: cv2.findChessboardCorners()를 통해 격자 패턴을 찾고, cv2.cornerSubPix()로 서브픽셀 단위의 정밀한 좌표를 획득합니다.

파라미터 추정: 3D 실제 좌표와 2D 이미지 좌표의 대응 관계를 이용해 cv2.calibrateCamera()로 내부 행렬(K)과 왜곡 계수를 계산합니다.

이미지 보정: 계산된 왜곡 계수를 cv2.undistort() 함수에 적용하여 렌즈의 배럴 왜곡 등을 제거합니다.

### 4. 출력 결과
![alt text](image-1.png)
![alt text](image.png)

## 02. 이미지 Rotation & Transformation
[Image Geometry & Affine Transformation]

### 1. 문제 설명

한 장의 원본 이미지에 대해 회전(Rotation), 크기 조절(Scaling), 평행이동(Translation)을 동시에 적용하는 아핀 변환을 실습합니다.

이미지 중심을 기준으로 +30도 회전, 0.8배 축소, 그리고 (x+80, y-40)만큼 이동시킨 결과를 출력합니다.

### 2. 코드

```Python
import cv2
import numpy as np

img = cv2.imread('rose.jpg') 
if img is not None:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 행렬 생성 (+30도, 0.8배)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    
    # 평행이동 값 반영 (x+80, y-40)
    M[0, 2] += 80
    M[1, 2] -= 40
    
    # 아핀 변환 적용
    dst = cv2.warpAffine(img, M, (w, h))
    
    combined = np.hstack((img, dst))
    cv2.imshow('Transformation Result', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
### 3. 해결 방법

변환 행렬 구성: cv2.getRotationMatrix2D()를 사용하여 회전과 스케일링이 포함된 2×3 변환 행렬을 생성합니다.

이동 연산 결합: 생성된 변환 행렬의 마지막 열(tx,ty) 값을 직접 수정하여 평행이동 효과를 추가합니다.

아핀 변환: cv2.warpAffine() 함수를 통해 기하학적 변환을 최종 이미지에 적용합니다.

### 4. 출력 결과
![alt text](image-2.png)

## 03. Stereo Disparity 기반 Depth 추정
[Stereo Vision & Depth Estimation]

### 1. 문제 설명

좌우 두 대의 카메라(Stereo Camera)에서 촬영된 이미지의 변위(Disparity)를 계산하여 물체와의 실제 거리(Depth)를 추정합니다.

특정 관심 영역(ROI - Painting, Frog, Teddy)에 대해 평균 Disparity와 Depth 값을 산출하고, 어떤 물체가 가장 가까운지 분석합니다.

### 2. 코드

```Python
import cv2
import numpy as np

left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

# 카메라 파라미터 (f: 초점거리, B: 베이스라인)
f, B = 700.0, 0.12
rois = {"Painting": (55, 50, 130, 110), "Frog": (90, 265, 230, 95), "Teddy": (310, 35, 115, 90)}

# 1. Disparity 계산
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# 2. Depth 계산 (Z = fB / d)
depth_map = np.zeros_like(disparity)
mask = disparity > 0
depth_map[mask] = (f * B) / disparity[mask]

# 3. ROI 분석
for name, (x, y, w, h) in rois.items():
    avg_depth = np.mean(depth_map[y:y+h, x:x+w][mask[y:y+h, x:x+w]])
    print(f"{name} Mean Depth: {avg_depth:.4f}m")

cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
```
### 3. 해결 방법

Disparity Map 생성: cv2.StereoBM_create()를 통해 좌우 영상의 픽셀 차이를 계산하고, 16배 스케일링된 정수 값을 실수로 보정합니다.

거리 산출 공식: 렌즈의 초점 거리(f)와 두 카메라 사이의 거리(B), 변위(d)를 이용하여

![alt text](image-5.png)
 
공식을 적용합니다.

영역 분석: ROI 슬라이싱을 통해 개별 물체의 거리를 측정하며, Disparity가 클수록 물체는 더 가깝다는 원리를 확인합니다.

### 4. 출력 결과
![alt text](image-4.png)
![alt text](image-3.png)