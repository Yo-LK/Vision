import cv2 # OpenCV 라이브러리 임포트
import numpy as np # 수치 계산을 위한 NumPy 라이브러리 임포트
import matplotlib.pyplot as plt # 시각화를 위한 Matplotlib 라이브러리 임포트

# 1. 두 개의 이미지 불러오기
img1_path = 'img1.jpg' # 왼쪽 기준이 될 이미지
img2_path = 'img2.jpg' # 변환되어 오른쪽에 붙을 이미지

img1 = cv2.imread(img1_path) # 왼쪽 기준이 될 이미지
img2 = cv2.imread(img2_path) # 변환되어 오른쪽에 붙을 이미지

if img1 is None or img2 is None: # 이미지가 제대로 불러와졌는지 확인
    print("오류: 이미지를 찾을 수 없습니다. 터미널 경로가 0326 폴더인지 확인해주세요.")
else:
    # Matplotlib 출력을 위해 BGR -> RGB 변환
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # SIFT 추출을 위해 흑백 이미지로 변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 객체 생성 및 특징점 추출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None) # img1에서 특징점과 디스크립터 추출
    kp2, des2 = sift.detectAndCompute(img2_gray, None) # img2에서 특징점과 디스크립터 추출

    # 3. 특징점 매칭 및 좋은 매칭점 선별
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 거리 비율이 임계값(0.7) 미만인 매칭점만 선별
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 매칭점 시각화 이미지 생성
    img_matching_result = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 호모그래피 계산을 위해서는 최소 4개의 매칭점이 필요합니다.
    if len(good_matches) >= 4:
        # 4. 호모그래피 행렬 계산
        # img2를 img1의 시점으로 변환하기 위해 src=img2, dst=img1로 설정
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # cv.RANSAC을 사용하여 이상점(Outlier) 영향 줄이기
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 5. 이미지 변환 및 정렬
        h1, w1 = img1_rgb.shape[:2] # img1의 높이와 너비
        h2, w2 = img2_rgb.shape[:2] # img2의 높이와 너비

        # 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1, h2))로 설정
        panorama_w = w1 + w2
        panorama_h = max(h1, h2)

        # img2를 호모그래피 행렬 H를 이용해 변환
        warped_img = cv2.warpPerspective(img2_rgb, H, (panorama_w, panorama_h))

        # 변환된 결과 캔버스 왼쪽에 원본 img1을 덮어씌워서 파노라마 완성
        warped_img[0:h1, 0:w1] = img1_rgb

        # 6. matplotlib을 이용하여 변환된 이미지와 매칭 결과 나란히 출력
        plt.figure(figsize=(18, 6))

        # 왼쪽: 매칭 결과 출력
        plt.subplot(1, 2, 1)
        plt.imshow(img_matching_result)
        plt.title('Matching Result')
        plt.axis('off')

        # 오른쪽: 호모그래피로 정합된 파노라마 이미지 출력
        plt.subplot(1, 2, 2)
        plt.imshow(warped_img)
        plt.title('Warped & Aligned Image (Panorama)')
        plt.axis('off')

        plt.tight_layout() # 서브플롯 간의 간격을 자동으로 조정하여 레이아웃을 깔끔하게 만듭니다.
        plt.show() # 그래프 창을 띄워서 결과를 시각화
    else:
        print(f"매칭점이 부족합니다. 최소 4개가 필요하지만 {len(good_matches)}개만 찾았습니다.")