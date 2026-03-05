import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # ROI 슬라이싱을 위한 numpy 임포트
import sys # 시스템 함수 사용을 위한 라이브러리 임포트

# 전역 변수 초기화
is_dragging = False # 드래그 상태 확인용 플래그
start_x, start_y = -1, -1 # 사각형 시작 좌표
roi = None # 잘라낸 이미지(ROI) 저장 변수

def select_roi(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global is_dragging, start_x, start_y, roi, img_copy # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 마우스 버튼 클릭 시작
        is_dragging = True # 드래그 상태 활성화
        start_x, start_y = x, y # 클릭한 시작 지점 좌표 저장

    elif event == cv.EVENT_MOUSEMOVE: # 마우스 드래그 중
        if is_dragging: # 드래그 상태일 때만 실행
            img_draw = img_copy.copy() # 잔상 방지를 위해 현재 이미지 복사본 생성
            # 빨간색(0, 0, 255)으로 두께 2인 사각형 그리기
            cv.rectangle(img_draw, (start_x, start_y), (x, y), (0, 0, 255), 2) 
            cv.imshow('ROI Selection', img_draw) # 드래그 중인 빨간 사각형을 실시간 출력

    elif event == cv.EVENT_LBUTTONUP: # 마우스 버튼을 놓았을 때
        is_dragging = False # 드래그 상태 비활성화
        x_min, x_max = min(start_x, x), max(start_x, x) # 좌우 방향 상관없이 x 범위 계산
        y_min, y_max = min(start_y, y), max(start_y, y) # 상하 방향 상관없이 y 범위 계산
        
        if x_max - x_min > 0 and y_max - y_min > 0: # 유효한 크기가 선택된 경우
            roi = img[y_min:y_max, x_min:x_max] # 원본에서 해당 영역 슬라이싱 추출
            cv.imshow('Extracted ROI', roi) # 추출된 ROI를 새 창에 표시
            # 선택 완료된 영역을 원본 복사본에 빨간 사각형으로 고정
            cv.rectangle(img_copy, (start_x, start_y), (x, y), (0, 0, 255), 2) 

# 이미지 로드 (soccer.jpg)
img = cv.imread('soccer.jpg') 

if img is None: # 이미지 파일이 없을 경우
    sys.exit('파일이 존재하지 않습니다') # 에러 메시지 출력 후 종료

img_copy = img.copy() # 원본 보존 및 드래그용 복사본 생성
cv.namedWindow('ROI Selection') # 윈도우 창 생성
cv.setMouseCallback('ROI Selection', select_roi) # 마우스 콜백 등록

print("--- 조작법 ---")
print("1. 드래그: 빨간 사각형으로 영역 선택")
print("2. r 키: 선택 영역 리셋")
print("3. s 키: 선택한 ROI 저장 (roi.jpg)")
print("4. q 키: 프로그램 종료")

while True: # 메인 루프 시작
    # 드래그 중이 아닐 때만 메인 화면 업데이트 (드래그 시의 실시간 출력을 방해하지 않음)
    if not is_dragging:
        cv.imshow('ROI Selection', img_copy) 
    
    key = cv.waitKey(1) & 0xFF # 1ms 대기하며 키 입력 인식

    if key == ord('r') or key == ord('R'): # 'r' 누를 시 초기화
        img_copy = img.copy() # 배경 이미지를 다시 원본으로 복구
        roi = None # 저장된 ROI 정보 삭제
        if cv.getWindowProperty('Extracted ROI', 0) >= 0: # 창이 열려있다면
            cv.destroyWindow('Extracted ROI') # ROI 창 닫기
        print("선택 영역이 리셋되었습니다.") 

    elif key == ord('s') or key == ord('S'): # 's' 누를 시 저장
        if roi is not None: # 선택된 ROI가 있다면
            cv.imwrite('roi.jpg', roi) # 파일로 저장
            print("선택 영역이 'roi.jpg'로 저장되었습니다.")
        else: # 선택된 ROI가 없다면
            print("저장할 영역이 선택되지 않았습니다.")

    elif key == ord('q') or key == ord('Q'): # 'q' 누를 시 종료
        break

cv.destroyAllWindows() # 모든 창 닫고 종료