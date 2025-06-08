import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
import pandas as pd
import sys

# ====== 설정 ======
image_path = "./images/starfield 1F.png"
coord_txt_path = "./txts/entrance_coordinates.txt"
map_path = "./txts/map_array.npy"
scale = 0.5

# ====== 맵 & 입구 방향 로딩 ======
map_array = np.load(map_path)
way_df = pd.read_excel("./cells/Class.cell")
number_to_way = dict(zip(way_df["Gate No"], way_df["Gate way"]))

# ====== 입구 좌표 불러오기 ======
entrances = []
with open(coord_txt_path, "r") as f:
    for line in f:
        if ":" in line:
            coord_str = line.strip().split(":")[1].strip(" ()\n")
            x, y = map(int, coord_str.split(","))
            entrances.append((x, y))

# ====== 좌표 변환 함수 ======
def to_map_coords(x, y): return int(x / scale), int(y / scale)
def to_gui_coords(x, y): return int(x * scale), int(y * scale)

# ====== 방향별 offset 설정 ======
def shifted_anchor(x, y, direction="down"):
    offsets = {"down": 80, "up": 80, "left": 40, "right": 40}
    offset = offsets.get(direction, 0)
    return {
        "down": (x, y + offset),
        "up": (x, y - offset),
        "left": (x + offset, y),
        "right": (x - offset, y),
    }.get(direction, (x, y))

# ====== 경로 탐색 BFS ======
def bfs_pathfinding(map_array, start, goal):
    h, w = map_array.shape
    visited = np.full((h, w), False)
    prev = np.full((h, w, 2), -1)
    queue = deque([start])
    visited[start[1], start[0]] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and map_array[ny, nx] == 1:
                visited[ny, nx] = True
                prev[ny, nx] = (x, y)
                queue.append((nx, ny))

    path = []
    x, y = goal
    if prev[y, x][0] == -1:
        return []
    while prev[y, x][0] != -1:
        path.append((x, y))
        x, y = prev[y, x]
    path.append(start)
    return path[::-1]

# ====== 이미지 로딩 ======
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
img_pil = Image.fromarray(img_resized)

# ====== GUI 초기화 ======
root = tk.Tk()
root.title("내 위치 & 경로 탐색 GUI")
canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
canvas.pack()
tk_img = ImageTk.PhotoImage(img_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# ====== 입구 표시 ======
for idx, (x, y) in enumerate(entrances):
    gx, gy = to_gui_coords(x, y)
    canvas.create_oval(gx - 5, gy - 5, gx + 5, gy + 5, fill="red", outline="black")
    canvas.create_text(gx, gy - 10, text=str(idx + 1), fill="black", font=("Arial", 7, "bold"))

# ====== 실행 시 전달된 Gate No 기반으로 위치 표시 ======
if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        anchor_nums = [int(x) for x in args if x.isdigit()]
        if not 1 <= len(anchor_nums) <= 4:
            raise ValueError("1~4개의 gate 번호 필요")

        coords = []
        for num in anchor_nums:
            if num < 1 or num > len(entrances):
                raise ValueError(f"입력된 gate 번호 {num}는 유효하지 않음")
            x, y = entrances[num - 1]
            direction = number_to_way.get(num, "down")
            coords.append(shifted_anchor(x, y, direction))

        unique_x = list(set(x for x, _ in coords))
        unique_y = list(set(y for _, y in coords))
        avg_x = sum(unique_x) // len(unique_x)
        avg_y = sum(unique_y) // len(unique_y)
        gx, gy = to_gui_coords(avg_x, avg_y)
        canvas.create_oval(gx - 15, gy - 10, gx + 15, gy + 10, outline="pink", fill="pink", width=1)
        canvas.create_text(gx, gy, text="나", fill="black", font=("Arial", 6, "bold"))
    except Exception as e:
        print("Gate No 처리 오류:", e)

root.mainloop()
