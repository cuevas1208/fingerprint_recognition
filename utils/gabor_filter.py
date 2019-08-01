"""
The principle of gabor filtering is to modify the value of the pixels of an image, generally in order to
improve its appearance. In practice, it is a matter of creating a new image using the pixel values
of the original image, in order to select in the Fourier domain the set of frequencies that make up
the region to be detected. The filter used is the Gabor filter with even symmetry and oriented at 0 degrees.

The resulting image will be the spatial convolution of the original (normalized) image and one of
the base filters in the direction and local frequency from the two directional and frequency maps
https://airccj.org/CSCP/vol7/csit76809.pdf pg.91
"""

import numpy as np
import scipy
def gabor_filter(im, orient, freq, kx=0.65, ky=0.65):
    """
    Gabor filter is a linear filter used for edge detection. Gabor filter can be viewed as a sinusoidal plane of
    particular frequency and orientation, modulated by a Gaussian envelope.
    :param im:
    :param orient:
    :param freq:
    :param kx:
    :param ky:
    :return:
    """
    angleInc = 3
    im = np.double(im)
    rows,cols = im.shape
    newim = np.zeros((rows,cols))
    
    freq_1d = np.reshape(freq,(1,rows*cols))
    ind = np.where(freq_1d>0)
    
    ind = np.array(ind)
    ind = ind[1,:]
    
    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100
    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    sigmax = 1/unfreq[0]*kx
    sigmay = 1/unfreq[0]*ky
    
    sze = np.round(3*np.max([sigmax,sigmay]))
    x,y = np.meshgrid(np.linspace(-sze,sze,(2*sze + 1)),np.linspace(-sze,sze,(2*sze + 1)))
    
    reffilter = np.exp(-(( (np.power(x,2))/(sigmax*sigmax) + (np.power(y,2))/(sigmay*sigmay)))) * np.cos(2*np.pi*unfreq[0]*x) # this is the original gabor filter
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180//angleInc,filt_rows,filt_cols)))
    
    for o in range(0,180//angleInc):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.
        rot_filt = scipy.ndimage.rotate(reffilter,-(o*angleInc + 90),reshape = False)
        gabor_filter[o] = rot_filt
                
    # Find indices of matrix points greater than maxsze from the image
    # boundary
    maxsze = int(sze)
    temp = freq>0
    validr,validc = np.where(temp)    
    
    temp1 = validr>maxsze
    temp2 = validr<rows - maxsze
    temp3 = validc>maxsze
    temp4 = validc<cols - maxsze
    
    final_temp = temp1 & temp2 & temp3 & temp4
    
    finalind = np.where(final_temp)
    
    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    maxorientindex = np.round(180/angleInc)
    orientindex = np.round(orient/np.pi*180/angleInc)
    
    # do the filtering
    for i in range(0,rows//16):
        for j in range(0,cols//16):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows,finalind_cols = np.shape(finalind)
    sze = int(sze)

    for k in range(0,finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]
        img_block = im[r-sze:r+sze + 1][:,c-sze:c+sze + 1]
        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r//16][c//16]) - 1])

    return(newim)
