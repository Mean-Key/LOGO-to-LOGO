import cv2
import numpy as np

img = cv2.imread("./images/starfield 1F.png")

# 흰색 영역 정의 (매장 내부)
white_mask = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))

# 검정색 영역 정의 (벽)
black_mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([30, 30, 30]))

# 이동 불가 영역 = 흰색 OR 검정
blocked = cv2.bitwise_or(white_mask, black_mask)

# 이동 가능 영역 = blocked == 0
map_array = (blocked == 0).astype(np.uint8)

# 저장
np.save("./txts/map_array.npy", map_array)