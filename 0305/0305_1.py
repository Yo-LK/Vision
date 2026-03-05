import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 배열 생성을 위한 numpy 임포트
import sys # 시스템 함수 사용을 위한 라이브러리 임포트

img = cv.imread('soccer.jpg')   # 이미지 로드

if img is None: # 정상적으로 로드되지 않은 경우
    sys.exit('파일이 존재하지 않습니다')    # 프로그램 종료
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환
gray_scale = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 그레이스케일 이미지를 BGR 형식으로 변환하여 원본 이미지와 같은 채널 수로 맞춤
combined = np.hstack((img, gray_scale)) # 원본 이미지와 그레이스케일 이미지를 가로로 연결

cv.imshow('Original and Gray Image Side-by-Side', combined) # 연결된 결과 이미지를 화면에 표시

cv.waitKey() # 키 입력 대기
cv.destroyAllWindows() # 모든 창 닫기