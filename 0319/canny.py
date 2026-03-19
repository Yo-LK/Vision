import cv2 # OpenCV 라이브러리
import numpy as np # 수치 계산을 위한 NumPy 라이브러리
from matplotlib import pyplot as plt # 시각화를 위한 Matplotlib 라이브러리

# 1. 이미지 불러오기
img = cv2.imread('0319/dabo.jpg')

if img is None: # 이미지가 제대로 불러와졌는지 확인
    print("이미지를 불러올 수 없습니다. 경로와 파일명을 확인하세요.")
else:
    # 원본 복사본 생성 (직선을 그릴 용도)
    line_img = img.copy()
    
    # 그레이스케일 변환 (에지 검출을 위한 전처리)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. cv.Canny()를 사용하여 에지 맵 생성
    edges = cv2.Canny(gray, 100, 200)

    # 3. cv.HoughLinesP()를 사용하여 직선 검출
    # 파라미터(rho, theta, threshold, minLineLength, maxLineGap)는 
    # 이미지 특성에 따라 조정이 필요하지만, 일반적인 기본값을 설정했습니다.
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=50, maxLineGap=10)

    # 4. cv.line()을 사용하여 검출된 직선을 원본 이미지에 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # HoughLinesP는 각 직선을 (x1, y1, x2, y2) 형태로 반환
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간색(0, 0, 255)으로 두께 2의 선을 그립니다.

    # 5. Matplotlib를 사용하여 시각화
    plt.figure(figsize=(12, 6))

    # 원본 이미지 (BGR -> RGB 변환)
    plt.subplot(1, 2, 1) # 원본 이미지 시각화
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV는 BGR 형식이므로 RGB로 변환하여 시각화
    plt.title('Original Image')
    plt.axis('off')

    # 직선이 그려진 결과 이미지 (BGR -> RGB 변환)
    plt.subplot(1, 2, 2) # 검출된 직선이 그려진 이미지 시각화
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)) # 검출된 직선이 그려진 이미지 시각화
    plt.title('Detected Lines (Hough Transform)')
    plt.axis('off')

    plt.tight_layout() # 그래프 간격 조정
    plt.show() # 시각화된 결과 출력