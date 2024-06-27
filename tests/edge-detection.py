"""
Video edge detection
"""
import cv2
import numpy as np

LOW_THRESHOLD_VALUE = 50
HIGH_THRESHOLD_VALUE = 150


def detect_edges(frame, low_threshold=LOW_THRESHOLD_VALUE, high_threshold=HIGH_THRESHOLD_VALUE):
    """
    Detect edges
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
    return edges


def process_video(video_path):
    """
    Process video
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Video not found at path: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        edges = detect_edges(frame)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        combined_frame = np.hstack((gray_frame, edges))
        
        cv2.imshow('Original and Edge-detected Video', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

video_path = '../videos/pink.mp4' 
process_video(video_path)
