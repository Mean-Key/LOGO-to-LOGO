import cv2
import numpy as np
import os

# ====== 층별 이미지 경로 ======
floors = [
    ("1F", "./images/etc/starfield_1F.png"),
    ("2F", "./images/etc/starfield_2F.png"),
    ("3F", "./images/etc/starfield_3F.png")
]

# ====== 출력 경로 ======
txt_output_path = "./txts/entrance_coordinates.txt"
image_output_dir = "./output"
os.makedirs("./txts", exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

# ====== 색상 범위 (BGR 기준) ======
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])

# ====== 공통 리스트 ======
all_points = []
counter = 1

# ====== 층별 처리 ======
for floor, image_path in floors:
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")
        continue

    print(f"[{floor}] 입구(빨간색) 인식 중...")

    # 빨간색 마스크 추출
    mask = cv2.inRange(img, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    entrance_points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        entrance_points.append((cx, cy))

    # 좌표 정렬
    entrance_points.sort(key=lambda pt: (-pt[1], -pt[0]))

    # 층별 좌표 추가 및 시각화
    for (x, y) in entrance_points:
        all_points.append((counter, floor, x, y))

        # 입구 표시 (빨간 원)
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

        # 게이트 번호 텍스트 (검은색, 점 바로 위에)
        text_position = (x - 10, y - 15)
        cv2.putText(
            img, str(counter), text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        counter += 1

    print(f"  → {len(entrance_points)}개 검출 완료")

    # 시각화 이미지 저장
    output_img_path = os.path.join(image_output_dir, f"starfield_{floor}_gates.png")
    cv2.imwrite(output_img_path, img)
    print(f"  → {floor} 게이트 표시 이미지 저장 완료: {output_img_path}")

# ====== 파일 저장 ======
with open(txt_output_path, "w") as f:
    for idx, floor, x, y in all_points:
        f.write(f"{idx}: ({x}, {y})  # {floor}\n")

print(f"\n✅ 모든 층의 입구 좌표 저장 완료 → {txt_output_path}")
print(f"✅ 게이트 표시 이미지 저장 완료 → {image_output_dir}/")
print(f"총 {len(all_points)}개 좌표가 저장되었습니다.")
