import cv2 # OpenCV 라이브러리 임포트
import numpy as np # 수치 계산을 위한 numpy 임포트
from pathlib import Path # 파일 경로 관리를 위한 pathlib 임포트

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True) # outputs 폴더가 없으면 생성

# 좌/우 이미지 불러오기 (파일 경로를 확인해 주세요)
left_color = cv2.imread("0312/left.png")
right_color = cv2.imread("0312/right.png")

if left_color is None or right_color is None: # 이미지 로드 실패 시 예외 처리
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다. '0312/left.png', '0312/right.png' 파일을 확인하세요.")

# 카메라 파라미터
f = 700.0  # focal length
B = 0.12   # baseline (m)

# ROI 설정 (x, y, w, h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# -----------------------------
# 그레이스케일 변환
# -----------------------------
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY) # StereoBM은 그레이스케일 입력 필요
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM 알고리즘 객체 생성
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# disparity map 계산 (16배 스케일된 정수형 반환)
disparity_raw = stereo.compute(left_gray, right_gray)

# 실수형 변환 후 16으로 나누어 실제 disparity 값 획득
disparity = disparity_raw.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산 (Z = fB / d)
# -----------------------------
depth_map = np.zeros_like(disparity) # 깊이 맵 초기화
valid_mask = disparity > 0  # 0보다 큰 픽셀만 사용

# 유효한 영역에 대해서만 Depth 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {} # ROI별 결과 저장 딕셔너리
for name, (x, y, w, h) in rois.items(): # 각 ROI에 대해
    roi_disp = disparity[y:y+h, x:x+w] # ROI 영역의 disparity 슬라이싱
    roi_depth = depth_map[y:y+h, x:x+w] # ROI 영역의 depth 슬라이싱
    roi_mask = valid_mask[y:y+h, x:x+w] # ROI 영역의 유효한 픽셀 마스크 (disparity > 0)
    
    # 유효한 픽셀에 대해서만 평균값 계산
    if np.any(roi_mask):
        mean_disp = np.mean(roi_disp[roi_mask]) # 유효한 disparity 픽셀의 평균
        mean_depth = np.mean(roi_depth[roi_mask]) # 유효한 depth 픽셀의 평균
    else:
        mean_disp, mean_depth = 0, 0 # 유효한 픽셀이 없는 경우 0으로 설정
        
    results[name] = {"mean_disp": mean_disp, "mean_depth": mean_depth} # 결과 저장

# -----------------------------
# 4. 결과 출력 및 해석
# -----------------------------
print(f"{'ROI Name':<10} | {'Mean Disparity':<15} | {'Mean Depth (m)':<15}") # 결과 표 헤더 출력
print("-" * 45)
for name, data in results.items(): # 각 ROI의 이름과 평균 disparity, 평균 depth 출력
    print(f"{name:<10} | {data['mean_disp']:<15.2f} | {data['mean_depth']:<15.4f}")

# 해석: Depth가 가장 작은 영역이 가장 가까움
closest_roi = min(results, key=lambda k: results[k]['mean_depth']) # Depth가 가장 작은 ROI 찾기
farthest_roi = max(results, key=lambda k: results[k]['mean_depth']) # Depth가 가장 큰 ROI 찾기

print(f"\n[해석] 가장 가까운 영역: {closest_roi} (Depth가 가장 작음)")
print(f"[해석] 가장 먼 영역: {farthest_roi} (Depth가 가장 큼)")

# -----------------------------
# 5. disparity 시각화 (제공된 템플릿 코드 사용)
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화 (제공된 템플릿 코드 사용)
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]
    z_min, z_max = np.percentile(depth_valid, 5), np.percentile(depth_valid, 95)
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = 1.0 - np.clip(depth_scaled, 0, 1) # 가까울수록 빨간색(높은값) 유도
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy() # ROI 표시를 위한 복사본 생성
for name, (x, y, w, h) in rois.items(): # 각 ROI에 대해
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2) # ROI 영역에 초록색 사각형 그리기
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # ROI 이름 텍스트 추가

# -----------------------------
# 8. 저장 및 출력
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color) # disparity 맵 저장
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color) # depth 맵 저장
cv2.imwrite(str(output_dir / "roi_result.png"), left_vis) # ROI 결과 이미지 저장

cv2.imshow("Disparity Map", disparity_color) # disparity 맵 시각화
cv2.imshow("Depth Map", depth_color) # depth 맵 시각화
cv2.imshow("ROI Marking", left_vis) # ROI 마킹 시각화
cv2.waitKey(0) # 키 입력 대기
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기