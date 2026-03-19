import cv2 # OpenCV 라이브러리
import numpy as np # 수치 계산을 위한 NumPy 라이브러리
from matplotlib import pyplot as plt # 시각화를 위한 Matplotlib 라이브러리
import matplotlib.colors as mcolors # 컬러맵 정의를 위한 Matplotlib의 colors 모듈

# ==============================================================================
# Edge and Region - 03 GrabCut을 이용한 대화식 영역 분할 및 객체 추출
# ==============================================================================

# * coffee cup 이미지로 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
# * 객체 추출 결과를 마스크 형태로 시각화
# * 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력

print("="*60)
print("Edge and Region")
print("03 GrabCut을 이용한 대화식 영역 분할 및 객체 추출 시작")
print("="*60)

# **[단계 1] 이미지 불러오기**
print("[단계 1] 이미지 불러오기...")
# img = cv2.imread('커피잔_이미지_파일_경로.jpg') # 사용자의 파일 경로
img = cv2.imread('0319/coffee_cup.jpg') # 생성한 시연용 이미지

if img is None: # 이미지가 제대로 불러와졌는지 확인
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# **[단계 2] GrabCut 초기 설정**
print("[단계 2] GrabCut 초기 설정...")
# 마스크를 원본 이미지와 같은 크기로 생성
mask = np.zeros(img.shape[:2], np.uint8) # GrabCut 알고리즘에서 사용할 마스크 (2D, uint8)
bgdModel = np.zeros((1, 65), np.float64) # 배경 모델 (1x65, float64)
fgdModel = np.zeros((1, 65), np.float64) # 전경 모델 (1x65, float64)

# **[단계 3] 초기 사각형 영역 설정**
print("[단계 3] 초기 사각형 영역 설정...")
# 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
# 시연용 이미지의 중앙 커피잔을 감싸는 영역
# (200, 300) 위치에서 (width=600, height=500) 크기
rect = (200, 300, 600, 500) 
print(f"설정된 사각형 영역 (x, y, width, height): {rect}")

# **[단계 4] cv.grabCut()을 사용하여 대화식 분할 수행**
print("[단계 4] cv.grabCut() 수행 (사각형 영역 기준)...")
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT) # 초기 사각형 영역을 기준으로 GrabCut 알고리즘 수행
print("GrabCut 알고리즘 수행 완료.")

# **[단계 5] 결과 마스크 처리 및 배경 제거**
print("[단계 5] 결과 마스크 처리 및 배경 제거...")
# GrabCut 수행 후 mask는 0~3 값을 가짐:
# 0: certain background, 1: certain foreground, 2: probable background, 3: probable foreground

# np.where()를 사용하여 마스크 값을 0 또는 1로 변경
# 확실한 전경(1)과 전경으로 추정되는 영역(3)은 1로, 나머지는 0으로 설정
# (mask==2) | (mask==0) 이면 0(배경), 아니면 1(객체)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
print("최종 객체 마스크(0/1) 생성 완료.")

# 마스크를 사용하여 원본 이미지에서 배경을 제거
# `numpy`의 broadcasting을 사용하여 원본 이미지에 최종 마스크 곱하기
# mask2는 2D이므로 3D로 확장하여 원본 이미지에 곱함
img_fg = img * mask2[:,:,np.newaxis]
print("원본 이미지에서 배경 제거 완료.")

# **[단계 6] 결과 시각화**
print("[단계 6] 결과 시각화...")

# 시각화를 위한 헬퍼 함수
def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 마스크 값을 시각적으로 명확하게 구분하기 위해 커스텀 컬러맵 정의
# 0: 검은색 (확실한 배경), 1: 노란색 (확실한 전경), 2: 파란색 (추정 배경), 3: 하늘색 (추정 전경)
colors = ['black', 'yellow', 'blue', 'cyan']
levels = [0, 1, 2, 3]
cmap_custom = mcolors.ListedColormap(colors) # 커스텀 컬러맵 생성
norm_custom = mcolors.BoundaryNorm(levels + [4], cmap_custom.N) # 마스크 값에 따른 컬러맵 정규화

plt.figure(figsize=(18, 6)) # 그래프 크기 설정

# matplotlib를 사용하여 세 개의 이미지를 나란히 시각화

# (1) 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(bgr_to_rgb(img))
plt.title('Original Image')
plt.axis('off')

# (2) GrabCut 마스크 (4단계 분류)
plt.subplot(1, 3, 2)
# 컬러맵을 사용하여 4가지 상태를 시각화
im_mask = plt.imshow(mask, cmap=cmap_custom, norm=norm_custom)
# 범례(colorbar) 추가
cbar = plt.colorbar(im_mask, ticks=[0, 1, 2, 3], orientation='horizontal', shrink=0.7) # 컬러바 추가
cbar.ax.set_xticklabels(['BGD (0)', 'FGD (1)', 'PR BGD (2)', 'PR FGD (3)']) # 컬러바 레이블 설정
plt.title('GrabCut Raw Mask (4 Levels)') # 마스크의 4단계 분류 시각화
plt.axis('off')

# (3) 배경 제거 이미지 (최종 객체 추출 결과)
plt.subplot(1, 3, 3) # 배경이 제거된 이미지 시각화
plt.imshow(bgr_to_rgb(img_fg)) # BGR -> RGB 변환
plt.title('Object Extraction (Background Removed)')
plt.axis('off')

plt.tight_layout()
print("="*60)
print("결과를 시각화했습니다. 그래프 창을 확인하세요.")
print("="*60)
plt.show()
