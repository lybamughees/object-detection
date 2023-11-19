import cv2
import time

def object_detection(video_path):
    # Load the car cascade classifier
    car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    start_time = time.time()  # Record the start time
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Measure execution time for object detection
        detection_start_time = time.time()
        
        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        detection_end_time = time.time()
        execution_time = detection_end_time - detection_start_time
        
        # Draw rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the frame with detections
        cv2.imshow('Car Detection', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    end_time = time.time()  # Record the end time
    total_execution_time = end_time - start_time
    
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection('sample_video.mp4')
