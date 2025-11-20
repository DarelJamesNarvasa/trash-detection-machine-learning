from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='FinalTrashScan.yolov8/data.yaml',
        epochs=10,
        imgsz=640,
        device='cpu',
        workers=0
    )

if __name__ == '__main__':
    main()