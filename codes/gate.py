import cv2
import numpy as np
import os


# 경로로
image_path = os.path.join("./images/starfild 1F.png")
output_path = os.path.join("./txts/entrance_coordinates.txt")


# ====== 이미지 로딩 ======
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# ====== 빨간색 범위 마스크 (BGR 기준) ======
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])
mask = cv2.inRange(img, lower_red, upper_red)

# ====== 윤곽선 기반 중심 좌표 추출 ======
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

entrance_points = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    entrance_points.append((cx, cy))

# ====== 좌표 정렬 (y 내림차순, x 내림차순) ======
entrance_points.sort(key=lambda pt: (-pt[1], -pt[0]))

# ====== 좌표 저장 ======
with open(output_path, "w") as f:
    for i, (x, y) in enumerate(entrance_points, 1):
        f.write(f"{i}: ({x}, {y})\n")

print(f"총 {len(entrance_points)}개 좌표 저장 완료 → {output_path}")
