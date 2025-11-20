import sys
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import heapq
from collections import deque

# ====== 인자 받기 ======
if len(sys.argv) < 8:
    print("사용법: find_different.py <start_x> <start_y> <start_floor> <goal_x> <goal_y> <goal_floor> <goal_name>")
    sys.exit(1)

start_x, start_y = map(int, sys.argv[1:3])
start_floor = sys.argv[3]
goal_x, goal_y = map(int, sys.argv[4:6])
goal_floor = sys.argv[6]
goal_name = sys.argv[7]

print(f"✅ 층간 이동 경로 탐색 시작: {start_floor} → {goal_floor}")
print(f"출발 좌표: ({start_x}, {start_y}) → 도착 좌표: ({goal_x}, {goal_y}) → 브랜드: {goal_name}")

# ====== 맵 / 이미지 로드 ======
def get_map_and_image(floor_label):
    map_path = f"./txts/map_array_{floor_label}.npy"
    img_path = f"./images/etc/starfield_{floor_label}.png"
    return np.load(map_path), cv2.imread(img_path)

map_start, img_start = get_map_and_image(start_floor)
map_goal, img_goal = get_map_and_image(goal_floor)

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
    dirs = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
        (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)
    ]

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

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
    h, w, _ = img.shape
    red_lower = np.array([0, 0, 200])
    red_upper = np.array([80, 80, 255])
    roi = img[max(0, y - search_radius):min(h, y + search_radius),
              max(0, x - search_radius):min(w, x + search_radius)]
    mask = cv2.inRange(roi, red_lower, red_upper)
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return x, y
    cx = int(M["m10"] / M["m00"]) - search_radius
    cy = int(M["m01"] / M["m00"]) - search_radius
    dir_vec = np.array([cx, cy], dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return x, y
    reverse_vec = -dir_vec / norm * offset
    rx, ry = int(x + reverse_vec[0]), int(y + reverse_vec[1])
    rx = np.clip(rx, 0, w - 1)
    ry = np.clip(ry, 0, h - 1)
    return rx, ry

# ====== 에스컬레이터 감지 ======
def detect_escalators(img):
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])
    mask = cv2.inRange(img, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    centers.sort(key=lambda p: (p[1], p[0]))
    return centers

# ====== 출발/도착 자동 보정 ======
start_x, start_y = find_nearest_walkable(map_start, start_x, start_y)
goal_x, goal_y = find_nearest_walkable(map_goal, goal_x, goal_y)

escalators_start = detect_escalators(img_start)
escalators_goal = detect_escalators(img_goal)

# ====== 올바른 에스컬레이터 쌍 선택 ======
if not escalators_start or not escalators_goal:
    print("⚠️ 에스컬레이터 감지 실패 → 기본값 사용")
    escalator_start = (500, 400)
    escalator_goal = (520, 380)
else:
    total_start = len(escalators_start)
    total_goal = len(escalators_goal)

    if (start_floor in ["1F", "2F"] and goal_floor == "3F") or \
       (start_floor == "3F" and goal_floor in ["1F", "2F"]):
        valid_indices = range(0, min(3, total_start, total_goal))
    else:
        valid_indices = range(0, min(6, total_start, total_goal))

    allowed_escalators = [escalators_start[i] for i in valid_indices]
    distances = [((e[0]-start_x)**2 + (e[1]-start_y)**2) for e in allowed_escalators]
    closest_index_in_allowed = int(np.argmin(distances))
    chosen_index = valid_indices[closest_index_in_allowed]

    escalator_start = escalators_start[chosen_index]
    escalator_goal  = escalators_goal[min(chosen_index, len(escalators_goal)-1)]
    print(f"🟩 에스컬레이터 쌍 선택: {chosen_index+1}번 ({escalator_start} → {escalator_goal})")

# ====== 경로 계산 (A*) ======
path_start = astar_path(map_start, (start_x, start_y), escalator_start)
path_goal  = astar_path(map_goal,  escalator_goal,  (goal_x, goal_y))
print(f"✅ 층간 이동 완료 (출발층 {len(path_start)} + 도착층 {len(path_goal)})")

# ====== Tkinter 윈도우 ======
root = tk.Tk()
root.title(f"층간 이동 경로 ({start_floor} ↔ {goal_floor})")

def to_tk_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    root.update_idletasks()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min(screen_w / img_pil.width * 0.7, screen_h / img_pil.height * 0.7)
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    img_resized = img_pil.resize(new_size)
    return ImageTk.PhotoImage(img_resized), scale

tk_start, scale_start = to_tk_image(img_start)
tk_goal, scale_goal   = to_tk_image(img_goal)

canvas = tk.Canvas(root, width=tk_start.width(), height=tk_start.height())
canvas.pack()

current_floor = [0]

def show_image(index):
    canvas.delete("all")
    if index == 0:
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_start)
        root.title(f"출발층: {start_floor}")
        for i in range(1, len(path_start)):
            x1, y1 = int(path_start[i-1][0] * scale_start), int(path_start[i-1][1] * scale_start)
            x2, y2 = int(path_start[i][0] * scale_start), int(path_start[i][1] * scale_start)
            canvas.create_line(x1, y1, x2, y2, fill="yellow", width=3, capstyle=tk.ROUND, smooth=True)
        sx, sy = int(start_x * scale_start), int(start_y * scale_start)
        canvas.create_oval(sx - 15, sy - 10, sx + 15, sy + 10, outline="pink", fill="pink")
        canvas.create_text(sx, sy, text="나", fill="black", font=("Arial", 9, "bold"))
    else:
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_goal)
        root.title(f"도착층: {goal_floor}")
        for i in range(1, len(path_goal)):
            x1, y1 = int(path_goal[i-1][0] * scale_goal), int(path_goal[i-1][1] * scale_goal)
            x2, y2 = int(path_goal[i][0] * scale_goal), int(path_goal[i][1] * scale_goal)
            canvas.create_line(x1, y1, x2, y2, fill="yellow", width=3, capstyle=tk.ROUND, smooth=True)
        rev_x, rev_y = get_reverse_anchor(img_goal, goal_x, goal_y)
        rev_x_scaled, rev_y_scaled = int(rev_x * scale_goal), int(rev_y * scale_goal)
        text_pad_x, text_pad_y = 8, 5
        text_w = 9 * max(1, len(goal_name))
        text_h = 18
        rx1, ry1 = rev_x_scaled - text_w // 2, rev_y_scaled - (text_h + 10)
        rx2, ry2 = rx1 + text_w + text_pad_x * 2, ry1 + text_h + text_pad_y * 2
        canvas.create_rectangle(rx1, ry1, rx2, ry2, fill="white", outline="blue")
        canvas.create_text((rx1 + rx2)//2, (ry1 + ry2)//2,
                           text=goal_name, fill="blue", font=("Arial", 10, "bold"))

def toggle_floor(event=None):
    current_floor[0] = 1 - current_floor[0]
    show_image(current_floor[0])

canvas.bind("<Button-1>", toggle_floor)
show_image(0)
root.mainloop()
