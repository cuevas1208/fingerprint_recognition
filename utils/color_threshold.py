from matplotlib import pyplot as plt
import cv2 as cv

def show_img_thresholds(img):
    ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_OTSU)
    ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY_INV','OTSU','BINARY|OTSU','BINARY','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    return thresh2
