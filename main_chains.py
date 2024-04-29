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
        images = [normalized_img, image]
        return self.successor.process((images, block_size)) if self.successor else normalized_img


class SegmentationHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size = data
        segmented_img, normim, mask = create_segmented_and_variance_images(images[0], block_size, 0.2)
        images.insert(0, normim)
        images.insert(0, segmented_img)
        return self.successor.process((images, block_size, mask)) if self.successor else segmented_img


class OrientationHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size, mask = data
        angles = orientation.calculate_angles(images[0], W=block_size, smoth=False)
        orientation_img = orientation.visualize_angles(images[0], mask, angles, W=block_size)
        images.insert(0, orientation_img)
        return self.successor.process((images, block_size, mask, angles)) if self.successor else orientation_img


class EnhancementHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size, mask, angles = data
        freq = ridge_freq(images[2], mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
        gabor_img = gabor_filter(images[2], angles, freq)
        images.insert(0, gabor_img)
        return self.successor.process((images, block_size, mask)) if self.successor else gabor_img


class SkeletonizationHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size, mask = data
        thin_image = skeletonize(images[0])
        images.insert(0, thin_image)
        return self.successor.process((images, block_size, mask)) if self.successor else thin_image


class MinutiaeHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size, mask = data
        minutiae_weights_image = calculate_minutiae_weights(images[0])
        images.insert(0, minutiae_weights_image)
        return self.successor.process((images, block_size, mask)) if self.successor else minutiae_weights_image


class BestRegionHandler(ImageProcessingHandler):
    def process(self, data):
        images, block_size, mask = data
        best_region = get_best_region(images[1], images[0], block_size, mask)
        result_image = draw_ridges_count_on_region(best_region, images[-1], images[1], block_size)
        images.insert(0, result_image)
        return images

block_size = 16
path = os.getcwd() + '/all_png_files'
img_name = "M89_f0115_03.png"
img_path = f'{path}/{img_name}'
image = cv2.imread(img_path, 0)
image = image[:-32, :]
cv2.imshow(img_name, image)
cv2.waitKey()

# Chains Definition
best_region_handler = BestRegionHandler()
minutiae_handler = MinutiaeHandler(best_region_handler)
skeletonization_handler = SkeletonizationHandler(minutiae_handler)
enhancement_handler = EnhancementHandler(skeletonization_handler)
orientation_handler = OrientationHandler(enhancement_handler)
segmentation_handler = SegmentationHandler(orientation_handler)
normalization_handler = NormalizationHandler(segmentation_handler)

# Getting images sequence
images = normalization_handler.process((image, block_size))
images.pop(1)
images.pop(-3)
labels = ['result', 'skeleton', 'gabor', 'orientation', 'segmented', 'normalized', 'original']
for step, (img, name) in enumerate(zip(images[::-1], labels[::-1])):
    cv2.imwrite(f'chain_{name}_step{step}.png', img)