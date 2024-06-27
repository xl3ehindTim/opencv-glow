import cv2
import numpy as np

# Load the image
image = cv2.imread('../../videos/image.png')

# Convert to YCrCb color space
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Define skin color range in YCrCb
lower_skin = np.array([0, 135, 85], dtype=np.uint8)
upper_skin = np.array([255, 180, 135], dtype=np.uint8)

# Threshold the YCrCb image to get only skin colors
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

# Use morphological operations to remove noise and fill gaps
kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)

# Blur the mask to help remove noise
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Get the bounding box for each contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # Optionally, filter out contours that are too small or too large
    if w > 20 and h > 20:
        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Detected Hand', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
