import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 배열 생성을 위한 numpy 임포트
import sys # 시스템 함수 사용을 위한 라이브러리 임포트

# 전역 변수 초기화
brush_size = 5 # 붓 크기 초기값
is_drawing = False # 마우스 드래그 상태 저장용
ix, iy = -1, -1 # 이전 마우스 좌표 저장용
color = (255, 0, 0) # 기본 색상 (파란색)

def draw(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global is_drawing, brush_size, color, ix, iy # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 클릭 시작
        is_drawing = True # 드래그 시작 상태로 변경
        color = (255, 0, 0) # 색상 설정 (파란색)
        ix, iy = x, y # 시작 좌표 저장
        cv.circle(img, (x, y), brush_size, color, -1) # 시작점 점 찍기

    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 클릭 시작
        is_drawing = True # 드래그 시작 상태로 변경
        color = (0, 0, 255) # 색상 설정 (빨간색)
        ix, iy = x, y # 시작 좌표 저장
        cv.circle(img, (x, y), brush_size, color, -1) # 시작점 점 찍기

    elif event == cv.EVENT_MOUSEMOVE: # 드래그 중
        if is_drawing:
            # 이전 좌표(ix, iy)에서 현재 좌표(x, y)까지 선을 그림
            # 선의 두께는 붓 크기의 2배로 설정하여 원의 지름과 맞춤
            cv.line(img, (ix, iy), (x, y), color, brush_size * 2) # 선 그리기
            ix, iy = x, y # 현재 좌표를 다음 선의 시작점으로 업데이트

    elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]: # 클릭 종료
        is_drawing = False # 드래그 종료 상태로 변경

img = cv.imread('soccer.jpg')   # 이미지 로드

if img is None: # 정상적으로 로드되지 않은 경우
    sys.exit('파일이 존재하지 않습니다')    # 프로그램 종료

cv.namedWindow('Painting on Soccer')   # 창 이름 설정
cv.setMouseCallback('Painting on Soccer', draw) # 마우스 이벤트 콜백 함수 등록

print("사용법: [=] 또는 [+] 크기 증가, [-] 크기 감소, [q] 종료")

while True: # 무한 루프를 돌며 이미지 표시 및 키 입력 대기
    cv.imshow('Painting on Soccer', img) # 이미지 표시
    
    key = cv.waitKey(1) & 0xFF # 키 입력 대기 및 하위 8비트만 추출

    # '+' 키는 Shift를 눌러야 하므로, 같은 키인 '='도 허용
    if key == ord('+') or key == ord('='): # 붓 크기 증가
        brush_size = min(15, brush_size + 1) # 최대 크기 15로 제한
        print(f"현재 붓 크기: {brush_size}") # 현재 붓 크기 출력
    
    elif key == ord('-'): # 붓 크기 감소
        brush_size = max(1, brush_size - 1) # 최소 크기 1로 제한
        print(f"현재 붓 크기: {brush_size}") # 현재 붓 크기 출력

    elif key == ord('q'): # 'q' 키를 누르면 루프 종료
        break # 루프 종료

cv.destroyAllWindows() # 모든 창 닫기