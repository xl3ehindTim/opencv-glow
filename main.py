import cv2

# Initialize video capture (replace 'video.mp4' with 0 to use a webcam)
cap = cv2.VideoCapture("./videos/greenonpurple.mp4")

# Check if the video file or webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Capture the first frame as the reference image
ret, reference_image = cap.read()

while True:
    # Capture frame-by-frame
    ret, current_frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("End of video stream or error.")
        break

    # Background subtraction
    diff = cv2.absdiff(reference_image, current_frame)

    # Thresholding
    _, thresh = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY)

    # Smoothing
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    output_frame = current_frame.copy()
    cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 3)

    # Display the result
    cv2.imshow('Detected Hand', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
