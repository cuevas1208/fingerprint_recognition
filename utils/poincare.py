from utils import orientation
import math
import cv2 as cv
import numpy as np

def poincare_index_at(i, j, angles, tolerance):
    """
    https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
    :param i:
    :param j:
    :param angles:
    :param tolerance:
    :return:
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5
    angles_around_index = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
    index = 0
    for k in range(0, 8):
        if abs(orientation.get_angle(angles_around_index[k], angles_around_index[k + 1])) > 90:
            angles_around_index[k + 1] += 180
        index += orientation.get_angle(angles_around_index[k], angles_around_index[k + 1])

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)}

    for i in range(3, len(angles) - 2):             # Y
        for j in range(3, len(angles[i]) - 2):      # x
            singularity = poincare_index_at(i, j, angles, tolerance)
            if singularity != "none":

                # mask any singularity outside of the mask
                mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
                mask_flag = np.sum(mask_slice)
                if mask_flag == (W*5)**2:
                    cv.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3)

    return result


if __name__ == '__main__':
    img = cv.imread('../test_img.png', 0)
    cv.imshow('original', img)
    angles = orientation.calculate_angles(img, 16, smoth=True)
    result = calculate_singularities(img, angles, 1, 16)
