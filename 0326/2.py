import cv2 # OpenCV 라이브러리 임포트
import matplotlib.pyplot as plt # 시각화를 위한 Matplotlib 라이브러리 임포트

# 1. 두 개의 이미지 불러오기 
img1_path = 'mot_color70.jpg'
img2_path = 'mot_color83.jpg'

img1 = cv2.imread(img1_path) # 왼쪽 기준이 될 이미지
img2 = cv2.imread(img2_path) # 변환되어 오른쪽에 붙을 이미지

# 파일이 제대로 불러와졌는지 확인
if img1 is None or img2 is None:
    print("오류: 이미지를 찾을 수 없습니다. 터미널 경로와 파일 이름을 다시 확인해주세요.")
else:
    # Matplotlib 출력을 위해 BGR -> RGB 변환
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # SIFT 추출을 위해 흑백 이미지로 변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 객체 생성 및 특징점 추출 
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # 3. 특징점 매칭 
    # knnMatch()를 사용하여 최근접 이웃 거리 비율 적용 (정확도 향상)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2) # 가장 가까운 매칭점 2개를 찾음

    # 좋은 매칭점만 선별하기 (Lowe's ratio test)
    # 첫 번째로 가까운 점의 거리가 두 번째로 가까운 점의 거리의 75% 이하일 때만 유효한 매칭으로 인정
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 4. 매칭 결과 시각화
    # drawMatches를 사용하여 두 이미지 간의 매칭 선을 그려줍니다.
    img_matches = cv2.drawMatches(
        img1_rgb, kp1, 
        img2_rgb, kp2, 
        good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # 매칭되지 않은 점은 그리지 않음
    )

    # 5. matplotlib을 이용한 매칭 결과 출력 (요구사항 반영)
    plt.figure(figsize=(15, 7))
    plt.imshow(img_matches)
    plt.title(f'SIFT Feature Matching (Good Matches: {len(good_matches)})')
    plt.axis('off')
    plt.tight_layout() # 서브플롯 간의 간격을 자동으로 조정하여 레이아웃을 깔끔하게 만듭니다.
    plt.show() # 그래프 창을 띄워서 결과를 시각화