import cv2
import time

# Storing input video file + model files
input_file = 'sample_video.mp4'
video_cap = cv2.VideoCapture(input_file)
model = 'haarcascade_car.xml'

# Using pre-defined model
carModel = cv2.CascadeClassifier(model)

start = time.time()

while True:
    
    # Starting to read video frames
    ret, frame = video_cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error reading video frame. Exiting...")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rescaling frame
    width = int(frame.shape[1] * 0.9)
    height = int(frame.shape[0] * 0.9)
    frame = cv2.resize(frame, (width, height))

    # Detecting cars
    detection_start = time.time()
    car_objects = carModel.detectMultiScale(gray_frame, 1.1, 7)
    detection_end = time.time()

    # Drawing boxes around cars
    for (x, y, w, h) in car_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'CAR', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # Displays GUI with running the video frame
    cv2.imshow('Detecting cars on the street...', frame)
    
    # Printing detection coordinates 
    print("Cars detected at: " + "[" + str(x) + ", " + str(y) + "]")
    
    # Calculating and printing time taken for detection
    detection_time = detection_end - detection_start
    print("Time taken for detection: {:.2f} seconds".format(detection_time))

    if cv2.waitKey(1) == 27:
        break
    
video_cap.release()
cv2.destroyAllWindows()
end = time.time()
print("Total Duration Time: {:.2f} seconds".format(end - start))
