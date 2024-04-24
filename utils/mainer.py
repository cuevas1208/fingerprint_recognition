# from main2 import FingerprintImageEnhancer
import os
import cv2

if __name__ == "__main__":
    print('doron')
    path = os.getcwd() + '/utils'
    img_name = "image.png"
    res = os.path.isfile(path + '/' + img_name)
    img_path = f'{path}/{img_name}'
    img = cv2.imread(img_path)
    cv2.imshow(img_name, img)
    print('what')
    # image_enhancer = FingerprintImageEnhancer()
    # enhanced_image = image_enhancer.enhance(img)

    # skeleton = skeletonize(enhanced_image)
    # minutiae_weights_image = calculate_minutiae_weights(thin_image)
    # block_size = 15
    # best_region = get_best_region(thin_image, minutiae_weights_image, block_size, mask)
    # result_image = draw_ridges_count_on_region(best_region, image, thin_image, block_size)
    #  result_image