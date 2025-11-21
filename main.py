from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='FinalTrashScan.yolov8/data.yaml',
        epochs=250,
        imgsz=640,
        # patience=50,
        # batch=16,
        # cache=True,
        # optimizer='AdamW',
        # lr=0.0005,
        # dropout=0.1,

        device='cpu',
        workers=0,
    )

if __name__ == '__main__':
    main()