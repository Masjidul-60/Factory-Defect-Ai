

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Tuple, Optional

# --- Configuration & Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [INFO] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    

    def __init__(self, image_path: str, output_dir: str = "processed_output"):
        """
        Initialize the processor with image path and output directory.
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = None
        self.processed_images = {}  # Dictionary to store intermediate results
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def load_image(self) -> bool:
        """Loads the image from disk and handles errors."""
        if not os.path.exists(self.image_path):
            logger.error(f"File not found: {self.image_path}")
            return False
            
        self.original_image = cv2.imread(self.image_path)
        
        if self.original_image is None:
            logger.error("Failed to load image. Check file format.")
            return False
            
        logger.info(f"Image loaded successfully: {self.image_path} | Resolution: {self.original_image.shape}")
        self.processed_images['Original'] = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        return True

    def resize_image(self, width: int = 800) -> None:
        """Resizes image to a standard width while maintaining aspect ratio."""
        h, w = self.original_image.shape[:2]
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio)
        
        resized = cv2.resize(self.original_image, (width, new_height), interpolation=cv2.INTER_AREA)
        self.original_image = resized
        # Update the stored RGB version
        self.processed_images['Original'] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        logger.info(f"Image resized to: {width}x{new_height}")

    def convert_to_grayscale(self) -> np.ndarray:
        """Converts the image to grayscale for processing."""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_images['Grayscale'] = gray
        return gray

    def apply_histogram_equalization(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Excellent for visualizing details in car shadows or metallic reflections.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray_img)
        self.processed_images['Equalized (Contrast)'] = equalized
        logger.info("Histogram Equalization (CLAHE) applied.")
        return equalized

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian Blur to reduce high-frequency noise.
        Essential step before Edge Detection.
        """
        # (5, 5) is the kernel size, 0 means sigma is calculated automatically
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        self.processed_images['Gaussian Blur'] = blurred
        logger.info("Noise reduction (Gaussian Blur) complete.")
        return blurred

    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a custom kernel to sharpen edges.
        """
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self.processed_images['Sharpened'] = sharpened
        logger.info("Sharpening filter applied.")
        return sharpened

    def detect_edges_canny(self, image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
        """
        Applies Canny Edge Detection.
        Double thresholding is used to detect strong and weak edges.
        """
        edges = cv2.Canny(image, low_thresh, high_thresh)
        self.processed_images['Canny Edges'] = edges
        logger.info(f"Canny Edge Detection applied (Thresholds: {low_thresh}, {high_thresh}).")
        return edges

    def apply_morphology(self, edges: np.ndarray) -> np.ndarray:
        """
        Applies Morphological Closing (Dilation followed by Erosion).
        This connects broken edge lines, useful for detecting whole scratches/cracks.
        """
        kernel = np.ones((3, 3), np.uint8)
        # Closing fills gaps inside object
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.processed_images['Morphology (Closed)'] = closed
        logger.info("Morphological Closing applied to connect edges.")
        return closed

    def find_and_draw_contours(self, binary_img: np.ndarray) -> None:
        """
        Finds contours based on the binary edge map and draws bounding boxes.
        This simulates 'Defect Detection' or 'Object Detection'.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the original to draw on
        result_img = self.original_image.copy()
        
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: Ignore very small noise, keep significant features
            if area > 100: 
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw Green Rectangle
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1
        
        # Convert BGR to RGB for Matplotlib
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        self.processed_images[f'Detected Features ({count})'] = result_rgb
        logger.info(f"Contour analysis complete. Found {count} significant features.")

    def save_processed_images(self) -> None:
        """Saves all generated images to the output directory."""
        for name, img in self.processed_images.items():
            # Clean filename
            filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".jpg"
            save_path = os.path.join(self.output_dir, filename)
            
            # Matplotlib images are RGB, OpenCV expects BGR
            if len(img.shape) == 3: # Color image
                save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else: # Grayscale
                save_img = img
                
            cv2.imwrite(save_path, save_img)
        logger.info(f"All processed images saved to '{self.output_dir}' folder.")

    def visualize_results(self) -> None:
        """
        Uses Matplotlib to display all processing steps in a grid layout.
        Dynamically adjusts based on how many images are in the dictionary.
        """
        num_images = len(self.processed_images)
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        plt.figure(figsize=(16, 5 * rows))
        
        for idx, (title, img) in enumerate(self.processed_images.items()):
            plt.subplot(rows, cols, idx + 1)
            
            if len(img.shape) == 2: # Grayscale
                plt.imshow(img, cmap='gray')
            else: # RGB
                plt.imshow(img)
                
            plt.title(title, fontsize=10, fontweight='bold')
            plt.axis('off')
            
        plt.tight_layout()
        plt.suptitle(f"Image Processing Analysis: {self.image_path}", fontsize=16)
        plt.show()
        logger.info("Visualization rendered.")

def main_execution_pipeline():
    """
    Main entry point for the script.
    """
    # 1. Define File Name
    TARGET_FILE = 'himel.jpg'
    
    # 2. Initialize Processor
    processor = AdvancedImageProcessor(TARGET_FILE)
    
    # 3. Load Image
    if not processor.load_image():
        return # Exit if load fails

    # 4. Resize (Optional, for performance on large images)
    processor.resize_image(width=800)

    # 5. Pipeline Execution
    try:
        # Step A: Convert to Gray
        gray = processor.convert_to_grayscale()
        
        # Step B: Contrast Enhancement (Good for cars)
        contrast = processor.apply_histogram_equalization(gray)
        
        # Step C: Noise Reduction (Prepare for edge detection)
        # We use the 'contrast' enhanced image for better edge detection
        blurred = processor.reduce_noise(contrast)
        
        # Step D: Sharpening (Optional visual enhancement)
        processor.apply_sharpening(blurred)
        
        # Step E: Edge Detection
        edges = processor.detect_edges_canny(blurred, low_thresh=50, high_thresh=150)
        
        # Step F: Morphological Operations (Refine edges)
        closed_edges = processor.apply_morphology(edges)
        
        # Step G: Feature/Defect Detection
        processor.find_and_draw_contours(closed_edges)
        
        # 6. Output Generation
        processor.save_processed_images()
        processor.visualize_results()
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}")
        raise e

if __name__ == "__main__":
    print("Starting Digital Image Processing Pipeline...")
    main_execution_pipeline()
    print("Processing Finished Successfully.")