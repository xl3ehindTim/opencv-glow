import cv2 
  
path = "videos/upstairs.mp4"

  
def preprocess(frame):
    """
    Preprocess frame with grayscale and blur
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def capture_video(path: str):
    """
    Capture video into opencv
    """
    return cv2.VideoCapture(path)


cap = capture_video(path)

# initializing subtractor  
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 


while(1): 
    ret, frame = cap.read()
   
    cv2.imshow('Original Video', frame)

    # Optional preprocessing
    # frame = preprocess(frame)

    # applying on each frame 
    fgmask = fgbg.apply(frame) 
  
    # cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow('frame', fgmask)   
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
cap.release() 
cv2.destroyAllWindows() 
