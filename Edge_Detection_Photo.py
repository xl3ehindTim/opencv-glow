import cv2
import numpy as np
from matplotlib import pyplot as plt

LowTresholdValue = 50
HightTresholdValue = 150

def detect_edges(image_path, low_threshold=LowTresholdValue, high_threshold=HightTresholdValue):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges

def show_images(original_image, edge_image):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(edge_image, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()

def main():
    image_path = './test/Images/TestImage2.0.png'  # Replace with your image path
    edges = detect_edges(image_path)
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    show_images(original_image, edges)

if __name__ == "__main__":
    main()