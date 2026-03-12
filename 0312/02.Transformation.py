import cv2 # OpenCV 라이브러리 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트

# 1. 이미지 로드
img = cv2.imread('0312/rose.png') 

if img is None:
    print("이미지를 찾을 수 없습니다. 파일명을 확인해 주세요.") # 이미지 로드 실패 메시지 출력
else:
    h, w = img.shape[:2] # 이미지 높이(h)와 너비(w) 추출
    
    # -----------------------------
    # 2. 변환 행렬 생성 (회전 & 크기 조절)
    # -----------------------------
    # 이미지의 중심점 계산
    center = (w // 2, h // 2)
    
    # cv2.getRotationMatrix2D(중심, 각도, 스케일)
    # +30도 회전, 0.8배 크기 조절
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    
    # -----------------------------
    # 3. 평행이동 반영 (Translation)
    # -----------------------------
    # 힌트: 회전 행렬의 마지막 열(tx, ty) 값을 조정
    # x축 방향으로 +80px, y축 방향으로 -40px 이동
    M[0, 2] += 80
    M[1, 2] -= 40
    
    # -----------------------------
    # 4. 아핀 변환 적용
    # -----------------------------
    # cv2.warpAffine(이미지, 변환행렬, 결과이미지크기)
    dst = cv2.warpAffine(img, M, (w, h))
    
    # -----------------------------
    # 5. 결과 시각화
    # -----------------------------
    # 원본과 결과를 가로로 붙여서 출력
    combined = np.hstack((img, dst))

    cv2.imshow('Image Transformation (Original vs Result)', combined) # 결과 시각화
    cv2.waitKey(0) # 키 입력 대기
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기