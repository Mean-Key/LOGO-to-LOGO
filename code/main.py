import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import sys
import os

# 실행할 스크립트 매핑
script_paths = {
    "Img": "YOLO_LOGO_img.py",
    "Video": "YOLO_LOGO_video.py",
    "Webcam": "YOLO_LOGO_webcam.py"
}

def run_script(name):
    if name == "Exit":
        root.destroy()
        sys.exit()
    elif name in script_paths:
        try:
            subprocess.Popen(["python", script_paths[name]])
        except Exception as e:
            print(f"{name} 실행 오류:", e)

# 메인 윈도우 설정
root = tk.Tk()
root.title("LOGO to LOGO")
root.geometry("1280x720")
root.configure(bg="black")

# ===== Header =====
header = tk.Label(root, text="LOGO to LOGO", font=("Helvetica", 24, "bold"),
                  fg="white", bg="black", anchor="center", justify="center")
header.pack(fill="x", pady=(10, 0))

# 헤더 구분선
header_line = tk.Frame(root, height=2, bg="white")
header_line.pack(fill="x", pady=(5, 0))

# ===== Body =====
body_frame = tk.Frame(root, bg="black")
body_frame.pack(expand=True, fill="both")

# 왼쪽 프레임 (3:1 비율)
left_frame = tk.Frame(body_frame, bg="black", width=int(1280 * 0.75), height=540)
left_frame.pack(side="left", fill="both", expand=True)

try:
    image = Image.open("./images/etc/title.png")
    image.thumbnail((940, 540))
    image_tk = ImageTk.PhotoImage(image)
    img_label = tk.Label(left_frame, image=image_tk, bg="black")
    img_label.image = image_tk
    img_label.pack(expand=True)
except Exception as e:
    tk.Label(left_frame, text="Image Load Error", fg="red", bg="black", font=("Helvetica", 14)).pack(expand=True)
    print("Image load error:", e)

# 세로 구분선
divider = tk.Frame(body_frame, width=2, bg="white")
divider.pack(side="left", fill="y")

# 오른쪽 프레임 (1/4 비율)
right_frame = tk.Frame(body_frame, bg="black", width=int(1280 * 0.25))
right_frame.pack(side="right", fill="y")

button_container = tk.Frame(right_frame, bg="black")
button_container.place(relx=0.5, rely=0.5, anchor="center")

button_texts = ["Img", "Video", "Webcam", "Exit"]
for text in button_texts:
    btn = tk.Button(button_container, text=text, width=20, height=2,
                    font=("Helvetica", 14), command=lambda t=text: run_script(t))
    btn.pack(pady=12)

# ===== Footer =====
footer_line = tk.Frame(root, height=2, bg="white")
footer_line.pack(fill="x", pady=(0, 5))

footer = tk.Label(root, text="Starfield HANAM", font=("Helvetica", 14), fg="white", bg="black", anchor="w")
footer.pack(fill="x", padx=20, pady=(0, 10))

root.mainloop()
