import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import subprocess
import sys
import cv2
from ultralytics import YOLO
from collections import Counter

# ====== 인자 확인 ======
if len(sys.argv) < 4:
    print("Usage: where.py <start_x> <start_y> <start_floor>")
    sys.exit(1)

start_x, start_y, start_floor = sys.argv[1:4]
model = YOLO("best.pt")

df = pd.read_excel("./cells/class.cell")
store_names = df["Name"].tolist()
gate_numbers = df["Gate No"].tolist()

root = tk.Tk()
root.title("목적지 선택")
root.geometry("900x700")

recognized_brand = tk.StringVar(value="")  # YOLO 인식 결과 저장
image_label = None  # 탐지 결과 이미지 표시용

# ====== 초기 화면 ======
def show_main_buttons():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="찾기 방식을 선택하세요", font=("Arial", 16, "bold")).pack(pady=40)
    tk.Button(root, text="이름으로 찾기", font=("Arial", 14), width=20, height=2,
              command=show_name_selector, bg="#ffffff").pack(pady=15)
    tk.Button(root, text="이미지로 찾기", font=("Arial", 14), width=20, height=2,
              command=show_image_search, bg="#ffffff").pack(pady=15)


# ====== 이름으로 찾기 ======
def show_name_selector():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="이동할 브랜드를 선택하세요", font=("Arial", 14, "bold")).pack(pady=10)

    # ====== 검색창 ======
    search_frame = tk.Frame(root)
    search_frame.pack(pady=5)

    search_var = tk.StringVar()

    search_entry = tk.Entry(search_frame, textvariable=search_var, font=("Arial", 12), width=40)
    search_entry.pack(side="left", padx=5)

    def update_list(event=None):
        """검색어 입력 시 리스트 필터링"""
        keyword = search_var.get().lower().strip()
        listbox.delete(0, tk.END)
        for name in store_names:
            if keyword in name.lower():
                listbox.insert(tk.END, name)

    search_entry.bind("<KeyRelease>", update_list)

    # ====== 리스트 ======
    listbox = tk.Listbox(root, font=("Arial", 12), height=20)
    listbox.pack(padx=10, pady=10, fill="both", expand=True)

    for name in store_names:
        listbox.insert(tk.END, name)

    selected_brand = tk.StringVar(value="")

    def on_select(event=None):
        """리스트 클릭 시 선택 저장"""
        selection = listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        brand_name = listbox.get(idx)
        selected_brand.set(brand_name)

    listbox.bind("<<ListboxSelect>>", on_select)

    # ====== 하단 버튼 영역 ======
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side="bottom", pady=10)

    # 선택 버튼
    def confirm_selection():
        brand_name = selected_brand.get()
        if not brand_name:
            tk.messagebox.showwarning("선택 필요", "브랜드를 선택해주세요.")
            return
        go_to_pathfinder(brand_name)

    select_btn = tk.Button(bottom_frame, text="선택", font=("Arial", 12, "bold"),
                           bg="#ffffff", width=10, command=confirm_selection)
    select_btn.pack(side="left", padx=5)

    # 뒤로가기 버튼
    back_btn = tk.Button(bottom_frame, text="뒤로가기", font=("Arial", 12, "bold"),
                         bg="#ffffff", width=10, command=show_main_buttons)
    back_btn.pack(side="left", padx=5)

    # 초기에 포커스 검색창으로
    search_entry.focus_set()

# ====== 이미지로 찾기 (YOLO 시각화 포함) ======
def show_image_search():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="이미지로 브랜드를 인식하세요", font=("Arial", 14, "bold")).pack(pady=10)

    tk.Button(root, text="이미지 선택", font=("Arial", 13), bg="#ffffff",
              command=select_image).pack(pady=5)

    # ====== 이미지 표시용 프레임 ======
    image_frame = tk.Frame(root, bg="white")
    image_frame.pack(fill="both", expand=True, padx=10, pady=5)

    global image_label
    image_label = tk.Label(image_frame, bg="white")
    image_label.pack(expand=True)

    result_label = tk.Label(root, textvariable=recognized_brand,
                            font=("Arial", 13, "bold"), fg="black")
    result_label.pack(pady=10)

    # ====== 하단 버튼 고정 ======
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side="bottom", pady=10, fill="x")

    tk.Button(bottom_frame, text="길찾기", font=("Arial", 14, "bold"),
              bg="#ffffff", command=start_finding).pack(side="left", expand=True, padx=20)

    tk.Button(bottom_frame, text="뒤로가기", font=("Arial", 14, "bold"),
              bg="#ffffff", command=show_main_buttons).pack(side="right", expand=   True, padx=20)


def select_image():
    """YOLO 이미지 인식 + output2.jpg 저장 + 화면 표시 (창 크기에 따라 자동 축소)"""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    results = model(img)[0]
    boxes = results.boxes
    filtered_boxes = boxes[boxes.conf >= 0.5]
    results.boxes = filtered_boxes

    # YOLO 결과 이미지 저장
    annotated = results.plot()
    cv2.imwrite("output2.jpg", annotated)

    # ====== 자동 크기 조정 ======
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    root.update_idletasks()
    frame_w = root.winfo_width()
    frame_h = root.winfo_height()

    # 버튼/여백 고려 (화면 비율 65% 이내 유지)
    max_w = int(frame_w * 0.75)
    max_h = int(frame_h * 0.55)

    pil_img.thumbnail((max_w, max_h))
    tk_img = ImageTk.PhotoImage(pil_img)

    image_label.config(image=tk_img)
    image_label.image = tk_img

    # 인식 결과 표시
    valid_classes = [int(box.cls[0]) for box in filtered_boxes]
    if not valid_classes:
        messagebox.showinfo("결과", "브랜드를 인식하지 못했습니다.")
        recognized_brand.set("인식 실패 ❌")
        return

    top_id = Counter(valid_classes).most_common(1)[0][0]
    brand_name = model.names[top_id]
    recognized_brand.set(f"인식된 브랜드: {brand_name}")
    print(f"인식된 브랜드: {brand_name}")


def start_finding():
    """길찾기 버튼 → finding.py로 이동"""
    brand_text = recognized_brand.get()
    if not brand_text or "인식된 브랜드:" not in brand_text:
        messagebox.showwarning("경고", "먼저 이미지를 인식해주세요.")
        return

    brand_name = brand_text.replace("인식된 브랜드:", "").strip()
    go_to_pathfinder(brand_name)


# ====== finding.py 호출 ======
def go_to_pathfinder(brand_name):
    brand_info = df[df["Name"] == brand_name]
    if brand_info.empty:
        messagebox.showerror("오류", "브랜드 정보를 찾을 수 없습니다.")
        return

    gate_no = int(brand_info["Gate No"].values[0])
    if 1 <= gate_no <= 100:
        goal_floor = "1F"
    elif 101 <= gate_no <= 178:
        goal_floor = "2F"
    else:
        goal_floor = "3F"

    # 좌표 찾기
    entrances = []
    with open("./txts/entrance_coordinates.txt", "r") as f:
        for line in f:
            if f"#{goal_floor}" in line.replace(" ", ""):
                coord_str = line.strip().split(":")[1].split("#")[0].strip(" ()\n")
                x, y = map(int, coord_str.split(","))
                entrances.append((x, y))

    # 층별 offset
    if goal_floor == "1F":
        offset = 0
    elif goal_floor == "2F":
        offset = 100
    elif goal_floor == "3F":
        offset = 178
    else:
        offset = 0

    idx = gate_no - offset - 1
    if idx < 0 or idx >= len(entrances):
        messagebox.showerror("오류", "목적지 좌표를 찾을 수 없습니다.")
        return

    goal_x, goal_y = entrances[idx]
    print(f"목적지 좌표: ({goal_x}, {goal_y}) / 층: {goal_floor}")

    # ====== 실행할 탐색 파일 선택 ======
    if start_floor == goal_floor:
        finding_file = "find_same.py"
    else:
        finding_file = "find_different.py"

    print(f"실행 파일: {finding_file}")

    # ====== 선택된 파일 실행 ======
    try:
        subprocess.call([
            sys.executable, finding_file,
            str(start_x), str(start_y), start_floor,
            str(goal_x), str(goal_y), goal_floor,
            brand_name   # 브랜드 이름 전달
        ])
        root.destroy()
    except Exception as e:
        messagebox.showerror("오류", f"{finding_file} 실행 중 문제 발생: {e}")

# ====== 프로그램 시작 ======
show_main_buttons()
root.mainloop()
