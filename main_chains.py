# from main2 import FingerprintImageEnhancer
import os
import cv2

from main_full import calculate_minutiae_weights, draw_ridges_count_on_region, get_best_region, skeletonize

from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation

if __name__ == "__main__":
    path = os.getcwd() + '/all_png_files'
    img_name = "M89_f0115_03.png"
    img_path = f'{path}/{img_name}'
    image = cv2.imread(img_path, 0)
    cv2.imshow(img_name, image)

    #Option 3 Alg (WON!!!):
    block_size = 16
    normalized_img = normalize(image.copy(), float(100), float(100))
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    gabor_img = gabor_filter(normim, angles, freq)
    enhanced_image = gabor_img

    thin_image = skeletonize(enhanced_image)
    minutiae_weights_image = calculate_minutiae_weights(thin_image)
    block_size = 15
    best_region = get_best_region(thin_image, minutiae_weights_image, block_size, mask)
    result_image = draw_ridges_count_on_region(best_region, image, thin_image, block_size)
    cv2.imshow('result_image', result_image)
    cv2.imwrite('result_image.png', result_image)
