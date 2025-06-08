import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import os
import pandas as pd
from collections import Counter
from ultralytics import YOLO
import subprocess
import sys

model = YOLO("best.pt")
df = pd.read_excel("./cells/class.cell")
df["Class ID"] = df["Class ID"].astype(str).str.upper().str.replace(" ", "")
brand_to_logo = dict(zip(df["Class ID"], df["Logo"]))

image_label = None
brand_labels = []
logo_labels = []
most_common_names = []

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if file_path:
        process_image(file_path)

def process_image(filepath):
    global most_common_names
    img = cv2.imread(filepath)
    if img is None:
        print("이미지 로드 실패")
        return

    results = model(img)[0]
    boxes = results.boxes
    filtered_boxes = boxes[boxes.conf >= 0.5]
    results.boxes = filtered_boxes

    annotated = results.plot()
    cv2.imwrite("output.jpg", annotated)

    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail((640, 480))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    valid_classes = [int(box.cls[0]) for box in filtered_boxes]
    top4 = Counter(valid_classes).most_common(4)
    most_common_names = [model.names[item[0]] for item in top4]

    for i in range(4):
        if i < len(top4):
            cls_id = top4[i][0]
            raw_name = model.names[cls_id]
            clean_name = raw_name.upper().replace(" ", "")
            brand_labels[i].config(text=raw_name)

            logo_filename = brand_to_logo.get(clean_name)
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

def show_location():
    # class.cell에서 Class ID → Class No → Gate No 매핑
    df_class = pd.read_excel("Class.cell")
    df_class["Class ID"] = df_class["Class ID"].astype(str).str.upper().str.replace(" ", "")
    df_class["Gate No"] = df_class["Gate No"].astype(str).str.strip()
    
    classid_to_classno = dict(zip(df_class["Class ID"], df_class["Class No"]))
    classno_to_gateno = dict(zip(df_class["Class No"], df_class["Gate No"]))

    gate_numbers = []
    for brand_name in most_common_names:
        clean_name = brand_name.upper().replace(" ", "")
        class_no = classid_to_classno.get(clean_name)
        gate_no = classno_to_gateno.get(class_no)
        if gate_no:
            try:
                gate_numbers.append(str(int(float(gate_no))))  # 소수점 처리 후 문자열 변환
            except:
                continue

    if 1 <= len(gate_numbers) <= 4:
        root.destroy()
        subprocess.call([sys.executable, "way.py", *gate_numbers])
    else:
        print("Gate No가 부족하여 위치를 계산할 수 없습니다.")

def back_to_main():
    root.destroy()  # 현재 창 닫기
    subprocess.call([sys.executable, "main.py"])  

root = tk.Tk()
root.title("YOLO Image Detection")
root.geometry("1280x720")
root.configure(bg="black")

# Header
tk.Label(root, text="FIND LOGO", font=("Helvetica", 24, "bold"), fg="white", bg="black").pack(fill="x", pady=(10, 0))
tk.Frame(root, height=2, bg="white").pack(fill="x", pady=(5, 0))

# Body
body = tk.Frame(root, bg="black")
body.pack(fill="both", expand=True)

# Left
left_frame = tk.Frame(body, bg="gray", width=640)
left_frame.pack(side="left", fill="both")
left_frame.pack_propagate(False)
image_label = tk.Label(left_frame, bg="white")
image_label.pack(expand=True, padx=10, pady=10)

# Middle
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

# Right
right_frame = tk.Frame(body, bg="black", width=320)
right_frame.pack(side="right", fill="y")
right_frame.pack_propagate(False)
button_container = tk.Frame(right_frame, bg="black")
button_container.place(relx=0.5, rely=0.5, anchor="center")

buttons = [
    ("이미지 선택", select_image),
    ("위치 보기", show_location),
    ("메인으로", back_to_main),
    ("종료", root.destroy)
]
for text, cmd in buttons:
    tk.Button(button_container, text=text, font=("Helvetica", 14), width=16, height=2, command=cmd).pack(pady=15)

# Footer
tk.Frame(root, height=2, bg="white").pack(fill="x", pady=(5, 0))
tk.Label(root, text="Starfield HANAM", font=("Helvetica", 12), fg="white", bg="black", anchor="w").pack(fill="x", padx=20, pady=10)

root.mainloop()
