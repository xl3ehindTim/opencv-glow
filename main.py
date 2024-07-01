import cv2

def init_video_capture(video_path):
    """
    Initializes video capture object.

    Args:
        video_path (str): Path to the video file or 0 for webcam.

    Returns:
        cv2.VideoCapture: The video capture object, or None if error occurs.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap


def capture_reference_frame(cap):
    """
    Captures the first frame as the reference image.

    Args:
        cap (cv2.VideoCapture): The video capture object.

    Returns:
        tuple: (success, reference_image)
    """
    ret, reference_image = cap.read()
    return ret, reference_image


def process_frame(reference_image, current_frame):
    """
    Processes a frame of the video.

    Args:
        reference_image (np.ndarray): The reference image.
        current_frame (np.ndarray): The current frame of the video.

    Returns:
        np.ndarray: The processed frame with detected contours.
    """
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

    return output_frame


def display_and_handle_key(frame, window_name):
    """
    Displays the frame and handles keyboard input.

    Args:
        frame (np.ndarray): The frame to display.
        window_name (str): The name of the window.
    """
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def release_resources(cap):
    """
    Releases the video capture object and closes OpenCV windows.

    Args:
        cap (cv2.VideoCapture): The video capture object.
    """
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    The main function that controls the video processing loop.
    """
    video_path = "./videos/greenonpurple.mp4"

    # Initialize video capture
    cap = init_video_capture(video_path)
    if not cap:
        return

    # Capture reference frame
    ret, reference_image = capture_reference_frame(cap)
    if not ret:
        return

    while True:
        # Capture frame-by-frame
        ret, current_frame = cap.read()

        # Handle end of stream or errors
        if not ret:
            print("End of video stream or error.")
            break

        # Process the frame
        processed_frame = process_frame(reference_image, current_frame)

        # Display the result
        display_and_handle_key(processed_frame, 'Detected Hand')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    release_resources(cap)


if __name__ == "__main__":
    main()