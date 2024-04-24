def count_fingerprint_ridges(image):
    print(image.shape)
    cropped_img = image[:-32,:]
    mask, normalized_img = normalize_and_segment(cropped_img)
    orientation = lines_orientation(normalized_img)
    frequency = calculate_ridge_frequencies(normalized_img, orientation, mask)
    enhanced_image = apply_gabor_filter(normalized_img, frequency, orientation)

    thin_image = skeletonize(enhanced_image)
    minutiae_weights_image = calculate_minutiae_weights(thin_image)

    block_size = 15
    best_region = get_best_region(thin_image, minutiae_weights_image, block_size, mask)
    result_image = draw_ridges_count_on_region(best_region, image, thin_image, block_size)
    return result_image