import os
import cv2
import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

# === 설정 ===
CONFIG = {
    'logo_dir': '.images/logo/',
    'background_dir': 'background/',
    'output_dir': 'dataset/',
    'num_per_class': 300,
    'logo_scale_range': (0.7, 1.1),
    'rotation_range': (-15, 15),
    'augment_each_logo': True,
    'image_size': (640, 640),
    'class_file': '.cells/class.cell'
}

def rotate_image_no_crop(img, angle):
    h, w = img.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255, 0))

def place_logo_on_background(bg, logo):
    h, w = bg.shape[:2]
    lh, lw = logo.shape[:2]
    if lw >= w or lh >= h:
        return bg, (0, 0, 0, 0)
    max_x = w - lw
    max_y = h - lh
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    overlay = bg.copy()
    alpha = logo[:, :, 3] / 255.0
    for c in range(3):
        overlay[y:y+lh, x:x+lw, c] = (
            alpha * logo[:, :, c] + (1 - alpha) * overlay[y:y+lh, x:x+lw, c]
        )
    x_center = (x + lw / 2) / w
    y_center = (y + lh / 2) / h
    w_norm = lw / w
    h_norm = lh / h
    return overlay, (x_center, y_center, w_norm, h_norm)

def generate():
    # 클래스명-클래스ID 로딩
    class_df = pd.read_excel(CONFIG['class_file'])
    name_to_id = dict(zip(class_df['Class Name'], class_df['Class ID']))

    # 로고 이미지 분류
    all_logos = glob(CONFIG['logo_dir'] + '*.png')
    class_to_images = {name: [] for name in name_to_id.keys()}

    for path in all_logos:
        filename = Path(path).stem.lower()
        for cls_name in name_to_id.keys():
            if cls_name.lower() in filename:
                class_to_images[cls_name].append(path)
                break

    # 출력 폴더
    out_img_dir = Path(CONFIG['output_dir']) / 'images'
    out_lbl_dir = Path(CONFIG['output_dir']) / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    bgs = glob(CONFIG['background_dir'] + '*.*')

    for cls_name, img_list in class_to_images.items():
        if not img_list:
            print(f"{cls_name}: 해당 클래스의 이미지가 없습니다.")
            continue

        class_id = name_to_id[cls_name]
        for i in range(CONFIG['num_per_class']):
            while True:
                bg = cv2.imread(random.choice(bgs))
                if bg is not None:
                    break
            bg = cv2.resize(bg, CONFIG['image_size'])

            logo_path = random.choice(img_list)
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None or logo.shape[2] != 4:
                continue

            scale = random.uniform(*CONFIG['logo_scale_range'])
            new_size = (int(logo.shape[1]*scale), int(logo.shape[0]*scale))
            logo = cv2.resize(logo, new_size, interpolation=cv2.INTER_AREA)

            if CONFIG['augment_each_logo']:
                angle = random.uniform(*CONFIG['rotation_range'])
                logo = rotate_image_no_crop(logo, angle)

            comp, label = place_logo_on_background(bg, logo)

            img_name = f"{cls_name}_{i:03d}.jpg"
            label_name = img_name.replace('.jpg', '.txt')
            cv2.imwrite(str(out_img_dir / img_name), comp)
            with open(out_lbl_dir / label_name, 'w') as f:
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in label)}\n")

if __name__ == '__main__':
    generate()
    print('✅ 데이터 생성 완료')
