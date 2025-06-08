import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    # 학습 시작
    model.train(
        data="C:/Users/HP_OMEN/Desktop/10/dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,          
        name='yolov8s_train_0605'
    )

if __name__ == "__main__":
    main()
