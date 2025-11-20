import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import sys
import re
import subprocess
from collections import Counter

# ====== 실행 시 전달된 Gate No 기반 ======
args = sys.argv[1:]
anchor_nums = [int(x) for x in args if x.isdigit()]

def get_floor_from_gate(gate_num):
    if 1 <= gate_num <= 100:
        return "1F"
    elif 101 <= gate_num <= 178:
        return "2F"
    else:
        return "3F"

# ====== 층 자동 선택 ======
floor_label = "1F"
image_path = "./images/etc/starfield_1F.png"

if anchor_nums:
    floors = [get_floor_from_gate(n) for n in anchor_nums]
    floor_label = Counter(floors).most_common(1)[0][0]
    image_path = f"./images/etc/starfield_{floor_label}.png"

print(f"✅ 선택된 층: {floor_label} / 이미지: {image_path}")

# ====== 기본 설정 ======
coord_txt_path = "./txts/entrance_coordinates.txt"
scale = 0.5

# ====== Class.cell 로드 및 정리 ======
way_df = pd.read_excel("./cells/class.cell")
way_df.columns = way_df.columns.str.strip()
way_df["Gate No"] = way_df["Gate No"].apply(lambda x: int(str(x).strip()) if pd.notna(x) else -1)
way_df["Name"] = way_df["Name"].fillna("").astype(str).str.strip()
number_to_way = dict(zip(way_df["Gate No"], way_df["Gate way"]))

# ====== 전역 게이트 맵: gate_no → (floor, x, y) ======
gate_map = {}
with open(coord_txt_path, "r") as f:
    for line in f:
        m = re.match(r"\s*(\d+)\s*:\s*\((\d+)\s*,\s*(\d+)\)\s*#\s*(\dF)", line)
        if not m:
            continue
        gno = int(m.group(1))
        x = int(m.group(2))
        y = int(m.group(3))
        fl = m.group(4)
        gate_map[gno] = (fl, x, y)

# ====== 현재 층 입구 좌표 목록 ======
entrances_current_floor = [(x, y) for g, (fl, x, y) in gate_map.items() if fl == floor_label]

# ====== 방향별 offset ======
def shifted_anchor(x, y, direction="down"):
    # 입구 방향에 따라 앵커 방향을 설정
    offsets = {"down": 80, "up": 80, "left": 50, "right": 50}
    offset = offsets.get(direction, 0)
    return {
        "down": (x, y + offset),
        "up": (x, y - offset),
        "left": (x - offset, y),
        "right": (x + offset, y),
    }.get(direction, (x, y))

# ====== 좌표 변환 ======
def to_gui_coords(x, y):
    return int(x * scale), int(y * scale)

# ====== GUI 초기화 ======
root = tk.Tk()
root.title(f"내 위치 ({floor_label})")

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
img_pil = Image.fromarray(img_resized)

canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
canvas.pack()
tk_img = ImageTk.PhotoImage(img_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# ====== 현재 층 입구 표시 (빨간 점) ======
for (x, y) in entrances_current_floor:
    gx, gy = to_gui_coords(x, y)
    canvas.create_oval(gx - 5, gy - 5, gx + 5, gy + 5, fill="red", outline="black")

# ====== 역앵커 및 브랜드명 표시 ======
label_positions = []  # 이미 그린 브랜드 이름 좌표를 저장 (겹침 방지용)

for gno in anchor_nums:
    if gno not in gate_map:
        continue
    fl, x, y = gate_map[gno]
    if fl != floor_label:
        continue  # 현재 층만 표시

    direction = number_to_way.get(gno, "down")
    reverse_dir = {"up": "down", "down": "up", "left": "right", "right": "left"}[direction]
    rx, ry = shifted_anchor(x, y, reverse_dir)

    # 브랜드 이름 (NaN/공백 안전 처리)
    brand_row = way_df.loc[way_df["Gate No"] == gno, "Name"]
    if not brand_row.empty:
        brand_name = brand_row.values[0]
        if not brand_name or brand_name.strip().lower() == "nan":
            brand_name = f"Gate {gno}"
    else:
        brand_name = f"Gate {gno}"

    rgx, rgy = to_gui_coords(rx, ry)
    label_y = rgy

    # ====== 겹침 방지 로직 ======
    for (px, py) in label_positions:
        dist = ((rgx - px) ** 2 + (label_y - py) ** 2) ** 0.5
        if dist < 20:  # 겹치면 35px 아래로 밀기
            label_y += 20

    label_positions.append((rgx, label_y))  # 위치 저장

    # ====== 브랜드명 전용 배경 박스 + 텍스트 ======
    font = ("Arial", 9, "bold")
    padding_x, padding_y = 6, 3

    tmp_id = canvas.create_text(rgx, label_y, text=brand_name, font=font)
    bbox = canvas.bbox(tmp_id)
    canvas.delete(tmp_id)

    if bbox:
        x1, y1, x2, y2 = bbox
        x1 -= padding_x
        y1 -= padding_y
        x2 += padding_x
        y2 += padding_y
        canvas.create_rectangle(x1, y1, x2, y2, fill="#FFFFFF", outline="black", width=1)

    # 텍스트 출력 (배경 위에)
    canvas.create_text(rgx, label_y, text=brand_name, fill="black", font=font)



# ====== 내 위치 계산 ======
selected_anchors = []
for gno in anchor_nums:
    if gno not in gate_map:
        continue
    fl, x, y = gate_map[gno]
    if fl != floor_label:
        continue
    direction = number_to_way.get(gno, "down")
    ax, ay = shifted_anchor(x, y, direction)
    selected_anchors.append((ax, ay))

if selected_anchors:
    avg_x = sum(x for x, _ in selected_anchors) // len(selected_anchors)
    avg_y = sum(y for _, y in selected_anchors) // len(selected_anchors)
elif entrances_current_floor:
    avg_x = sum(x for x, _ in entrances_current_floor) // len(entrances_current_floor)
    avg_y = sum(y for _, y in entrances_current_floor) // len(entrances_current_floor)
else:
    h, w, _ = img.shape
    avg_x, avg_y = w // 2, h // 2

gx, gy = to_gui_coords(avg_x, avg_y)
canvas.create_oval(gx - 15, gy - 10, gx + 15, gy + 10, outline="pink", fill="pink")
canvas.create_text(gx, gy, text="나", fill="black", font=("Arial", 8, "bold"))

# ====== 길찾기 버튼 ======
def where():
    """길찾기 버튼 클릭 시 where.py 실행"""
    try:
        subprocess.call([
            sys.executable, "where.py",
            str(avg_x), str(avg_y), floor_label
        ])
    except Exception as e:
        print("목적지 선택 실행 오류:", e)

path_button = tk.Button(root, text="길찾기", font=("Arial", 12, "bold"),
                        bg="#FFFFFF", command=where)
path_button.pack(pady=10)

root.mainloop()
