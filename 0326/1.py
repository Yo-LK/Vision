import cv2 # OpenCV 라이브러리 임포트
import matplotlib.pyplot as plt # 시각화를 위한 Matplotlib 라이브러리 임포트

# 1. 이미지 불러오기
image_path = 'mot_color70.jpg' 
img = cv2.imread(image_path)

if img is None: # 이미지가 제대로 불러와졌는지 확인
    print(f"오류: '{image_path}' 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    # OpenCV는 BGR 형태로 이미지를 읽으므로 Matplotlib 출력을 위해 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 특징점 검출을 위해 흑백 이미지로 변환 (필수는 아니지만 일반적으로 연산 속도와 효율을 위해 사용)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 객체 생성 
    sift = cv2.SIFT_create()

    # 3. 특징점 검출 및 특징 디스크립터 계산
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    # 4. 특징점 시각화 
    img_keypoints = cv2.drawKeypoints(
        img_rgb, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 5. matplotlib을 이용한 결과 출력 
    plt.figure(figsize=(15, 7)) # 전체 창 크기 설정

    # 왼쪽: 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 오른쪽: 특징점이 시각화된 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(img_keypoints)
    plt.title(f'SIFT Keypoints (Total: {len(keypoints)})')
    plt.axis('off')

    plt.tight_layout() # 서브플롯 간의 간격을 자동으로 조정하여 레이아웃을 깔끔하게 만듭니다.
    plt.show() # 그래프 창을 띄워서 결과를 시각화