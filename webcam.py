from ultralytics import YOLO

# 1. Load your trained model
model = YOLO('runs/detect/train4/weights/best.pt')

# 2. Set up the prediction stream
# stream=True allows us to process the video one frame at a time
results = model.predict(source=0, stream=True, show=True, conf=0.5)

print("Starting camera... Press 'q' in the camera window to exit.")

# 3. Loop through every frame from the camera
for result in results:
    # Get all the boxes detected in this frame
    boxes = result.boxes

    # Count them using len() (length)
    count = len(boxes)

    # Print the count to the console
    if count > 0:
        print(f"Trash bags detected: {count}")
    else:
        print("No trash detected.")