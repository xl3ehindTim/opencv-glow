import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

image_path = './test/Images/Roze.png'
reference_image_path = './test/Images/Reference.png'
LowTresholdValue = 50
HightTresholdValue = 150

def detect_edges(image_path, low_threshold=LowTresholdValue, high_threshold=HightTresholdValue):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    
    # Apply bilateral filter to reduce noise and keep edges sharp
    image = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges

def show_images(original_image, edge_image, reference_image):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(132), plt.imshow(edge_image, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(133), plt.imshow(reference_image, cmap='gray')
    plt.title('Reference Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()

def main():
    edges = detect_edges(image_path)
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute SSIM between edge image and reference image
    ssim_value, _ = ssim(edges, reference_image, full=True)
    print(f"SSIM between edge image and reference image: {ssim_value:.4f}")
    
    show_images(original_image, edges, reference_image)

if __name__ == "__main__":
    main()