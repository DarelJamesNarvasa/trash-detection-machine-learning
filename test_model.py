from ultralytics import YOLO

# 1. Load the model you just trained
# Note: We point to 'train4' because that's what your screenshot showed
model = YOLO('runs/detect/train4/weights/best.pt')

# 2. Run a prediction on the "test" folder of images
# This will take all images in the test folder and draw boxes on them
results = model.predict(
    source='FinalTrashScan.yolov8/test/images',
    save=True,      # Save the images with boxes drawn
    conf=0.25       # Only show detections if AI is 25% sure
)

print("Testing complete! Check the runs/detect/predict folder.")