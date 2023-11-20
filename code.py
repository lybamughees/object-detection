import cv2
import time

def obstacle_detection(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Load pre-trained object detection model (Haarcascades for simplicity)
    cascade_path = "haarcascade_car.xml"
    obstacle_cascade = cv2.CascadeClassifier(cascade_path)

    start_time = time.time()

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect obstacles
        obstacles = obstacle_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around detected obstacles
        for (x, y, w, h) in obstacles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Obstacle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken for obstacle detection: {elapsed_time:.2f} seconds")

    # Release video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_video_path.mp4' with the path to your video file
    obstacle_detection('sample_video.mp4')

