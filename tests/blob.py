"""
Hand detection using blobs
"""
import cv2


def detect_light_video(video_path, threshold=127):
  """
  Detects light regions in a video using blob detection

  Args:
      video_path: Path to the video file
      threshold: Grayscale threshold for light detection (default 127)
  """

  cap = cv2.VideoCapture(video_path)

  while (1):
    ret, frame = cap.read()

    if not ret:
      print("No frames captured from video. Breaking loop...")
      break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    # Find contours (blobs)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Draw contours on original frame (optional)
    for c in cnts:
      cv2.drawContours(frame, [c], 0, (0, 255, 0), 2)

    # Display the frame with detected light regions
    cv2.imshow("Light Detection (Video)", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

  # Release capture
  cap.release()
  cv2.destroyAllWindows()

video_path = "../videos/greenonpurple.mp4"
detect_light_video(video_path)
