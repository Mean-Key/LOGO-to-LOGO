import sys
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import heapq
from collections import deque

# ====== 인자 받기 ======
if len(sys.argv) < 8:
    print("사용법: find_same.py <start_x> <start_y> <start_floor> <goal_x> <goal_y> <goal_floor> <goal_name>")
    sys.exit(1)

start_x, start_y = map(int, sys.argv[1:3])
start_floor = sys.argv[3]
goal_x, goal_y = map(int, sys.argv[4:6])
goal_floor = sys.argv[6]
goal_name = sys.argv[7]

# ====== 맵 / 이미지 로드 ======
map_path = f"./txts/map_array_{start_floor}.npy"
img_path = f"./images/etc/starfield_{start_floor}.png"
map_array = np.load(map_path)
img = cv2.imread(img_path)

# ====== 주변 통로(1)로 자동 보정 ======
def find_nearest_walkable(map_array, x, y, radius=15):
    h, w = map_array.shape
    for r in range(1, radius):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if map_array[ny, nx] == 1:
                        return nx, ny
    return x, y

# ====== A* 탐색 (8방향) ======
def astar_path(map_array, start, goal):
    h, w = map_array.shape

    # 8방향 (대각선 포함)
    dirs = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
        (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)
    ]

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])  # 유클리드 거리

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            break

        for dx, dy, move_cost in dirs:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if map_array[int(ny), int(nx)] != 1:
                continue

            new_cost = g_score[current] + move_cost
            neighbor = (nx, ny)
            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, new_cost, neighbor))
                came_from[neighbor] = current

    # 경로 복원
    path = []
    node = goal
    if node not in came_from:
        return [start]
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

# ====== 역앵커 계산 ======
def get_reverse_anchor(img, x, y, search_radius=10, offset=25):
    """
    주어진 입구 좌표(x, y)에서 빨간색(입구 방향)을 찾아
    반대 방향(offset 픽셀)으로 역앵커 좌표를 반환.
    """
    h, w, _ = img.shape
    red_lower = np.array([0, 0, 200])
    red_upper = np.array([80, 80, 255])
    roi = img[max(0, y - search_radius):min(h, y + search_radius),
              max(0, x - search_radius):min(w, x + search_radius)]
    mask = cv2.inRange(roi, red_lower, red_upper)
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return x, y  # 빨간색 없으면 그대로 반환

    # 입구 방향 벡터 (입구 중심 → 빨간색 중심)
    cx = int(M["m10"] / M["m00"]) - search_radius
    cy = int(M["m01"] / M["m00"]) - search_radius
    dir_vec = np.array([cx, cy], dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return x, y

    # 역방향으로 offset 만큼 이동
    reverse_vec = -dir_vec / norm * offset
    rx, ry = int(x + reverse_vec[0]), int(y + reverse_vec[1])

    rx = np.clip(rx, 0, w - 1)
    ry = np.clip(ry, 0, h - 1)
    return rx, ry

# ====== 출발/도착 자동 보정 ======
start_x, start_y = find_nearest_walkable(map_array, start_x, start_y)
goal_x, goal_y = find_nearest_walkable(map_array, goal_x, goal_y)

# ====== 경로 계산 ======
path = astar_path(map_array, (start_x, start_y), (goal_x, goal_y))
print(f"✅ A* (8방향) 경로 계산 완료: {len(path)} 스텝")

# ====== Tkinter 윈도우 ======
root = tk.Tk()
root.title(f"경로 탐색 ({start_floor})")

# ====== 배경 이미지 표시 ======
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

root.update_idletasks()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
scale = min((screen_w * 0.7) / img_pil.width, (screen_h * 0.7) / img_pil.height)
scale = max(0.4, min(scale, 1.0))
new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
img_pil = img_pil.resize(new_size, Image.LANCZOS)
tk_img = ImageTk.PhotoImage(img_pil)

canvas = tk.Canvas(root, width=tk_img.width(), height=tk_img.height())
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# ====== 좌표 스케일 ======
img_h, img_w = img.shape[:2]
scale_x = tk_img.width() / img_w
scale_y = tk_img.height() / img_h

# ====== 경로 표시 ======
if len(path) > 1:
    for i in range(1, len(path)):
        x1, y1 = int(path[i-1][0] * scale_x), int(path[i-1][1] * scale_y)
        x2, y2 = int(path[i][0] * scale_x), int(path[i][1] * scale_y)
        canvas.create_line(x1, y1, x2, y2, fill="yellow", width=3, capstyle=tk.ROUND, smooth=True)
else:
    print("⚠️ 경로 없음 (스텝 1)")

# ====== 출발점 ("나") ======
sx, sy = int(start_x * scale_x), int(start_y * scale_y)
canvas.create_oval(sx - 15, sy - 10, sx + 15, sy + 10, outline="pink", fill="pink")
canvas.create_text(sx, sy, text="나", fill="black", font=("Arial", 9, "bold"))

# ====== 도착점 (브랜드 이름 - 역앵커 기반) ======
rev_x, rev_y = get_reverse_anchor(img, goal_x, goal_y)
rev_x_scaled, rev_y_scaled = int(rev_x * scale_x), int(rev_y * scale_y)

text_pad_x, text_pad_y = 8, 5
text_w = 9 * max(1, len(goal_name))
text_h = 18
rx1, ry1 = rev_x_scaled - text_w // 2, rev_y_scaled - (text_h + 10)
rx2, ry2 = rx1 + text_w + text_pad_x * 2, ry1 + text_h + text_pad_y * 2

canvas.create_rectangle(rx1, ry1, rx2, ry2, fill="white", outline="blue")
canvas.create_text((rx1 + rx2)//2, (ry1 + ry2)//2,
                   text=goal_name, fill="blue", font=("Arial", 10, "bold"))

root.mainloop()
