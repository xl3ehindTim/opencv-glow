"""
Background substractor MOG2 test
"""
import cv2 
  
cap = cv2.VideoCapture("./videos/greenonpurple.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2() 
  
  
def preprocess(frame):
    """
    Preprocess frame with grayscale and blur
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


while True: 
    ret, frame = cap.read() 
  
    frame = preprocess(frame)

    fgmask = fgbg.apply(frame) 

    cv2.imshow('fgmask', fgmask) 
    cv2.imshow('frame',frame ) 
      
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
      
  
cap.release() 
cv2.destroyAllWindows() 