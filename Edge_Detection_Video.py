import cv2
import numpy as np

LowTresholdValue = 50
HightTresholdValue = 150

def detect_edges(frame, low_threshold=LowTresholdValue, high_threshold=HightTresholdValue):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, low_threshold, high_threshold)
    return edges

def process_video(video_path):
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

def main():
    video_path = './test/Video/RozeLicht.mp4'  # Replace with your video path
    process_video(video_path)

if __name__ == "__main__":
    main()