from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

LOGO_DIR = "logos/"
OUTPUT_DIR = "warped_logos/"
DIRECTIONS = ['top', 'bottom', 'left', 'right']
RATIOS = [10, 17, 25]

def clean_image(path):
    # Pillow로 열고 다시 저장해서 메타데이터 제거
    img = Image.open(path).convert("RGBA")
    clean_path = path.with_name(f"clean_{path.name}")
    img.save(clean_path, format="PNG")
    return clean_path

def apply_perspective(img, direction, ratio):
    h, w = img.shape[:2]
    delta = int(min(h, w) * (ratio / 100))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    if direction == 'top':
        dst = np.float32([[delta, 0], [w - delta, 0], [w, h], [0, h]])
    elif direction == 'bottom':
        dst = np.float32([[0, 0], [w, 0], [w - delta, h], [delta, h]])
    elif direction == 'left':
        dst = np.float32([[0, delta], [w, 0], [w, h], [0, h - delta]])
    else:
        dst = np.float32([[0, 0], [w, delta], [w, h - delta], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255, 0))

def generate():
    logo_paths = list(Path(LOGO_DIR).glob("*.png"))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for logo_path in logo_paths:
        cleaned_path = clean_image(logo_path)
        logo_name = logo_path.stem
        logo = cv2.imread(str(cleaned_path), cv2.IMREAD_UNCHANGED)

        for direction in DIRECTIONS:
            for ratio in RATIOS:
                warped = apply_perspective(logo, direction, ratio)
                outname = f"{logo_name}_{direction}_{ratio}.png"
                cv2.imwrite(str(Path(OUTPUT_DIR) / outname), warped)

        cleaned_path.unlink()  # 중간 파일 삭제

    print("왜곡된 로고 생성 완료")

if __name__ == '__main__':
    generate()
