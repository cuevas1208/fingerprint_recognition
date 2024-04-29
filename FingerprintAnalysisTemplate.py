from abc import ABC, abstractmethod
import numpy as np
import cv2
from skimage.morphology import skeletonize as ski_skeletonize
import os
from main_full import calculate_minutiae_weights, draw_ridges_count_on_region, get_best_region, skeletonize
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation

class FingerprintAnalysisTemplate(ABC):

    def __init__(self, block_size=16):
        self.block_size = block_size

    def analyze_fingerprint(self, image):
        normalized_img = self.normalize(image)
        segmented_img, normim, mask = self.segment(normalized_img)
        # should I save the orientation_img for later visualization?
        angles, orientation_img = self.calculate_orientation(normim, segmented_img, mask)
        freq = self.calculate_frequency(normim, mask, angles)
        gabor_filtered_img = self.apply_gabor_filters(normim, angles, freq)
        thin_image = self.skeletonize(gabor_filtered_img)
        minutiae_weights_image = self.calculate_minutiae_weights(thin_image)
        best_region = self.select_best_region(thin_image, minutiae_weights_image, mask)
        result_image = self.draw_ridges_count_on_region(best_region, image, thin_image)
        return [image, normalized_img, segmented_img, orientation_img, gabor_filtered_img, thin_image, result_image]
        
    @abstractmethod
    def normalize(self, image):
        pass

    @abstractmethod
    def segment(self, normalized_img):
        pass

    @abstractmethod
    def calculate_orientation(self, normim, segmented_img, mask):
        pass

    @abstractmethod
    def calculate_frequency(self, normim, mask, angles):
        pass

    @abstractmethod
    def apply_gabor_filters(self, normim, angles, freq):
        pass

    @abstractmethod
    def skeletonize(self, gabor_filtered_img):
        pass

    @abstractmethod
    def calculate_minutiae_weights(self, thin_image):
        pass

    @abstractmethod
    def select_best_region(self, thin_image, minutiae_weights_image, mask):
        pass

    @abstractmethod
    def draw_ridges_count_on_region(self, best_region, image, thin_image):
        pass

class ConcreteFingerprintAnalysis(FingerprintAnalysisTemplate):

    def analyze_fingerprint(self, image):
        return super().analyze_fingerprint(image)

    def normalize(self, image):
        return normalize(image.copy(), float(100), float(100))
    
    def segment(self, normalized_img):
        return create_segmented_and_variance_images(normalized_img, self.block_size, 0.2)
    
    def calculate_orientation(self, image, segmented_img, mask):
        angles = orientation.calculate_angles(image, W=self.block_size, smoth=False)
        orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=self.block_size)

        return angles, orientation_img
        
    def calculate_frequency(self, normim, mask, angles):
        return ridge_freq(normim, mask, angles, self.block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    
    def apply_gabor_filters(self, normim, angles, freq):
        return gabor_filter(normim, angles, freq)
    
    def skeletonize(self, gabor_filtered_img):
        return skeletonize(gabor_filtered_img)
    
    def calculate_minutiae_weights(self, thin_image):
        return calculate_minutiae_weights(thin_image)
    
    def select_best_region(self, thin_image, minutiae_weights_image, mask):
        return get_best_region(thin_image, minutiae_weights_image, self.block_size, mask)
    
    def draw_ridges_count_on_region(self, best_region, image, thin_image):
        return draw_ridges_count_on_region(best_region, image, thin_image, self.block_size)
    
def client_code(fingerprint_analysis: FingerprintAnalysisTemplate, image):
    return fingerprint_analysis.analyze_fingerprint(image)

if __name__ == '__main__':
    input_path = './all_png_files/'
    output_path = './all_png_files_out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = os.getcwd() + '/all_png_files'
    img_name = "M89_f0115_03.png"
    img_path = f'{path}/{img_name}'
    image = cv2.imread(img_path, 0)
    image = image[:-32, :]

    cv2.imshow(img_name, image)
    cv2.waitKey()

    images = client_code(ConcreteFingerprintAnalysis(), image)
    lables = ['image', 'normalized_img', 'segmented_img', 'orientation_img', 'gabor_filtered_img', 'skeletonize_img', 'result_image']

    for step, (img, name) in enumerate(zip(images, lables)):
        cv2.imwrite(f'template_{name}_step{step}.png', img)