import cv2
import numpy as np

# Load the reference image (empty tube) and the current frame
reference_image = cv2.imread("./videos/reference_red.png")
current_frame = cv2.imread("./videos/image.png")

# # Convert images to grayscale
# gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
# gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

# Background subtraction
diff = cv2.absdiff(reference_image, current_frame)

# Thresholding
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

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
cv2.imshow('DetectedHand_v', output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
