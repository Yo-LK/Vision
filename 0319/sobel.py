import cv2 # OpenCV 라이브러리
import numpy as np # 수치 계산을 위한 NumPy 라이브러리
from matplotlib import pyplot as plt # 시각화를 위한 Matplotlib 라이브러리

# 1. cv.imread()를 사용하여 이미지 불러오기
img = cv2.imread('0319/edgeDetectionImage.jpg')

if img is None: # 이미지가 제대로 불러와졌는지 확인
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    # 2. cv.cvtColor()를 사용하여 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. cv.Sobel()을 사용하여 x축과 y축 방향의 에지 검출
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # x축 방향의 에지 검출
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # y축 방향의 에지 검출

    # 4. cv.magnitude()를 사용하여 에지 강도(magnitude) 계산
    magnitude = cv2.magnitude(sobelx, sobely)

    # 5. 힌트: cv.convertScaleAbs()를 사용하여 uint8로 변환
    # 시각화를 위해 절대값을 취하고 8비트 형식으로 바꾼다.
    sobel_abs = cv2.convertScaleAbs(magnitude)

    # 6. Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
    plt.figure(figsize=(12, 6))

    # 원본 이미지 출력 (RGB로 변환 필요)
    plt.subplot(1, 2, 1)  # 원본 이미지 시각화
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 에지 강도 이미지 출력 (그레이스케일로 시각화)
    plt.subplot(1, 2, 2)  # 에지 강도 이미지 시각화
    plt.imshow(sobel_abs, cmap='gray') # 에지 강도 이미지는 그레이스케일로 시각화
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')

    plt.tight_layout() # 그래프 간격 조정
    plt.show() # 시각화된 결과 출력