import os
import cv2
from main_full import calculate_minutiae_weights, draw_ridges_count_on_region, get_best_region, skeletonize
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation

class ImageProcessingHandler:
    def __init__(self, successor=None):
        self.successor = successor

    def process(self, data):
        pass


class NormalizationHandler(ImageProcessingHandler):
    def process(self, data):
        image, block_size = data
        normalized_img = normalize(image.copy(), float(100), float(100))
        return self.successor.process((normalized_img, block_size)) if self.successor else normalized_img


class SegmentationHandler(ImageProcessingHandler):
    def process(self, data):
        image, block_size = data
        segmented_img, normim, mask = create_segmented_and_variance_images(image, block_size, 0.2)
        return self.successor.process((segmented_img, block_size, mask, normim)) if self.successor else segmented_img


class OrientationHandler(ImageProcessingHandler):
    def process(self, data):
        segmented_img, block_size, mask, normin = data
        angles = orientation.calculate_angles(segmented_img, W=block_size, smoth=False)
        orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
        return self.successor.process((orientation_img, block_size, mask, normin, angles)) if self.successor else orientation_img


class EnhancementHandler(ImageProcessingHandler):
    def process(self, data):
        orientation_img, block_size, mask, normin, angles = data
        freq = ridge_freq(orientation_img, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
        gabor_img = gabor_filter(orientation_img, angles, freq)
        return self.successor.process((gabor_img, block_size, mask, angles)) if self.successor else gabor_img


class SkeletonizationHandler(ImageProcessingHandler):
    def process(self, data):
        gabor_img, block_size, mask, angles = data
        thin_image = skeletonize(gabor_img)
        return self.successor.process((thin_image, block_size, mask, angles)) if self.successor else thin_image


class MinutiaeHandler(ImageProcessingHandler):
    def process(self, data):
        thin_image, block_size, mask, angles = data
        minutiae_weights_image = calculate_minutiae_weights(thin_image)
        return self.successor.process((minutiae_weights_image, block_size, mask, angles)) if self.successor else minutiae_weights_image


class BestRegionHandler(ImageProcessingHandler):
    def process(self, data):
        minutiae_weights_image, block_size, mask, angles = data
        best_region = get_best_region(minutiae_weights_image, block_size, mask)
        result_image = draw_ridges_count_on_region(best_region, mask, block_size)
        return result_image

# Processing the image
block_size = 16
path = os.getcwd() + '/all_png_files'
img_name = "M89_f0115_03.png"
img_path = f'{path}/{img_name}'
image = cv2.imread(img_path, 0)
cv2.imshow(img_name, image)

# Creating the chain
best_region_handler = BestRegionHandler()
minutiae_handler = MinutiaeHandler(best_region_handler)
skeletonization_handler = SkeletonizationHandler(minutiae_handler)
enhancement_handler = EnhancementHandler(skeletonization_handler)
orientation_handler = OrientationHandler(enhancement_handler)
segmentation_handler = SegmentationHandler(orientation_handler)
normalization_handler = NormalizationHandler(segmentation_handler)

# Processing the image
result_image = normalization_handler.process((image, block_size))