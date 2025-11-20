import cv2
import numpy as np
import os

# ====== 층별 이미지 경로 ======
floors = [
    ("1F", "./images/etc/starfield_1F.png"),
    ("2F", "./images/etc/starfield_2F.png"),
    ("3F", "./images/etc/starfield_3F.png")
]

# ====== 출력 폴더 ======
output_dir = "./txts"
os.makedirs(output_dir, exist_ok=True)

# ====== 색상 범위 정의 (BGR 기준) ======
color_ranges = {
    "WHITE": ([240, 240, 240], [255, 255, 255]),  # 매장 내부
    "BLACK": ([0, 0, 0], [30, 30, 30]),           # 벽
    "GRAY": ([100, 100, 100], [180, 180, 180]),   # 이동 가능 구역
    "GREEN": ([0, 150, 0], [100, 255, 100]),      # 에스컬레이터
    "RED": ([0, 0, 200], [50, 50, 255])           # 입구
}

# ====== 층별 처리 ======
for floor, image_path in floors:
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")
        continue

    print(f"[{floor}] 색상 영역 인식 중...")

    # 개별 마스크 생성
    masks = {}
    for color_name, (low, high) in color_ranges.items():
        masks[color_name] = cv2.inRange(img, np.array(low), np.array(high))
        print(f"  {color_name:<6} 검출 완료")

    # 이동 불가 = 흰색 + 검정 + 빨강
    blocked = cv2.bitwise_or(masks["WHITE"], masks["BLACK"])
    blocked = cv2.bitwise_or(blocked, masks["RED"])

    # 이동 가능 = 회색 + 초록
    walkable = cv2.bitwise_or(masks["GRAY"], masks["GREEN"])

    # 최종 맵 (1 = 이동 가능, 0 = 이동 불가능)
    map_array = np.where(walkable > 0, 1, 0).astype(np.uint8)

    # numpy 파일로 저장
    map_path = os.path.join(output_dir, f"map_array_{floor}.npy")
    np.save(map_path, map_array)

    print(f"  ✅ {floor} 맵 저장 완료 → {map_path}")

print("\n✅ 모든 층의 맵 인식 및 저장 완료!")
