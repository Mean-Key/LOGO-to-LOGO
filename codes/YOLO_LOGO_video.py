# YOLO_LOGO_video.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import os
import pandas as pd
from collections import Counter
from ultralytics import YOLO
import subprocess
import threading
import sys

model = YOLO("best.pt")
df = pd.read_excel("./cells/class.cell")
df["Class ID"] = df["Class ID"].astype(str).str.strip().str.upper()
brand_to_logo = dict(zip(df["Class ID"], df["Logo"]))

image_label = None
brand_labels = []
logo_labels = []
video_running = False
cap = None
brand_counter = Counter()
MIN_OCCURRENCE_COUNT = 50

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        threading.Thread(target=process_video, args=(file_path,), daemon=True).start()

def process_video(filepath):
    global cap, video_running, brand_counter
    cap = cv2.VideoCapture(filepath)
    video_running = True
    brand_counter.clear()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter("annotated_result.mp4", fourcc, fps, (w, h))

    while video_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes
        filtered_boxes = boxes[boxes.conf >= 0.5]
        results.boxes = filtered_boxes

        annotated = results.plot()
        video_writer.write(annotated)
        show_frame(annotated)

        valid_classes = [int(box.cls[0]) for box in filtered_boxes]
        brand_counter.update(valid_classes)

        filtered_top = [item for item in brand_counter.items() if item[1] >= MIN_OCCURRENCE_COUNT]
        filtered_top.sort(key=lambda x: -x[1])
        top4 = filtered_top[:4]

        for i in range(4):
            if i < len(top4):
                cls_id = top4[i][0]
                brand_name = model.names[cls_id].strip().upper()
                brand_labels[i].config(text=brand_name)

                logo_filename = brand_to_logo.get(brand_name)
                if logo_filename:
                    logo_path = os.path.join("./images/logo", logo_filename)
                    if os.path.exists(logo_path):
                        try:
                            logo_img = Image.open(logo_path).resize((120, 60))
                            logo_img = ImageOps.expand(logo_img, border=1, fill='gray')
                            logo_tk = ImageTk.PhotoImage(logo_img)
                            logo_labels[i].config(image=logo_tk)
                            logo_labels[i].image = logo_tk
                        except:
                            logo_labels[i].config(image='')
                    else:
                        logo_labels[i].config(image='')
                else:
                    logo_labels[i].config(image='')
            else:
                brand_labels[i].config(text="")
                logo_labels[i].config(image='')

    cap.release()
    video_writer.release()

def show_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail((640, 480))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

def play_result_video():
    output_path = "annotated_result.mp4"
    if os.path.exists(output_path):
        subprocess.Popen(["start", output_path], shell=True)
    else:
        print("저장된 결과 영상이 없습니다.")

def show_location():
    # class.cell에서 Class ID → Class No → Gate No 매핑
    df_class = pd.read_excel("./cells/class.cell")
    df_class["Class ID"] = df_class["Class ID"].astype(str).str.upper().str.replace(" ", "")
    df_class["Gate No"] = df_class["Gate No"].astype(str).str.strip()

    classid_to_classno = dict(zip(df_class["Class ID"], df_class["Class No"]))
    classno_to_gateno = dict(zip(df_class["Class No"], df_class["Gate No"]))

    gate_numbers = []
    # 누적된 브랜드 중 상위 4개 가져오기 (이미 MIN_OCCURRENCE_COUNT로 필터됨)
    filtered_top = [item for item in brand_counter.items() if item[1] >= MIN_OCCURRENCE_COUNT]
    filtered_top.sort(key=lambda x: -x[1])
    top4 = filtered_top[:4]

    for cls_id, _ in top4:
        raw_name = model.names[cls_id]
        clean_name = raw_name.upper().replace(" ", "")
        class_no = classid_to_classno.get(clean_name)
        gate_no = classno_to_gateno.get(class_no)
        if gate_no:
            try:
                gate_numbers.append(str(int(float(gate_no))))
            except:
                continue

    if 1 <= len(gate_numbers) <= 4:
        root.destroy()
        subprocess.call([sys.executable, "way.py", *gate_numbers])
    else:
        print("Gate가 부족하여 위치를 계산할 수 없습니다.")


def back_to_main():
    global video_running
    video_running = False
    root.destroy()
    subprocess.call([sys.executable, "main.py"])

def stop():
    global video_running
    video_running = False
    root.destroy()

# GUI
root = tk.Tk()
root.title("YOLO Video Detection")
root.geometry("1280x720")
root.configure(bg="black")

tk.Label(root, text="FIND LOGO", font=("Helvetica", 24, "bold"), fg="white", bg="black").pack(fill="x", pady=(10, 0))
tk.Frame(root, height=2, bg="white").pack(fill="x", pady=(5, 0))

body = tk.Frame(root, bg="black")
body.pack(fill="both", expand=True)

left_frame = tk.Frame(body, bg="gray", width=640)
left_frame.pack(side="left", fill="both")
left_frame.pack_propagate(False)
image_label = tk.Label(left_frame, bg="white")
image_label.pack(expand=True, padx=10, pady=10)

middle_frame = tk.Frame(body, bg="white", width=320)
middle_frame.pack(side="left", fill="y")
middle_frame.pack_propagate(False)
tk.Label(middle_frame, text="Brands:", font=("Helvetica", 16, "bold"), bg="white", fg="black").pack(pady=(20, 0))
tk.Frame(middle_frame, height=2, bg="black").pack(fill="x", pady=(2, 10))

for _ in range(4):
    brand = tk.Label(middle_frame, text="", font=("Helvetica", 14), fg="black", bg="white")
    brand.pack(pady=(10, 2))
    logo = tk.Label(middle_frame, bg="white")
    logo.pack(pady=(0, 10))
    brand_labels.append(brand)
    logo_labels.append(logo)

right_frame = tk.Frame(body, bg="black", width=320)
right_frame.pack(side="right", fill="y")
right_frame.pack_propagate(False)
button_container = tk.Frame(right_frame, bg="black")
button_container.place(relx=0.5, rely=0.5, anchor="center")

buttons = [
    ("동영상 선택", select_video),
    ("위치 보기", show_location),
    ("결과 보기", play_result_video),
    ("메인으로", back_to_main),
    ("종료", stop)
]
for text, cmd in buttons:
    tk.Button(button_container, text=text, font=("Helvetica", 14), width=16, height=2, command=cmd).pack(pady=15)

tk.Frame(root, height=2, bg="white").pack(fill="x", pady=(5, 0))
tk.Label(root, text="Starfield HANAM", font=("Helvetica", 12), fg="white", bg="black", anchor="w").pack(fill="x", padx=20, pady=10)

root.mainloop()
