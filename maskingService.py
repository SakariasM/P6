# Python program to illustrate foreground extraction using GrabCut algorithm
# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
def mask_data(input_image):
    image = cv2.imread(input_image)

    # Get image dimensions
    height, width = image.shape[:2]
    
    # create a simple mask image similar
    # to the loaded image, with the 
    # shape and return type
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # specify the background and foreground model
    # using numpy the array is constructed of 1 row
    # and 65 columns, and all array elements are 0
    # Data type for the array is np.float64 (default)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    
    # define the Region of Interest (ROI)
    # automatically calculated based on image size
    # Using 10% margin from edges
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    rect_width = width - 2 * margin_x
    rect_height = height - 2 * margin_y
    rectangle = (margin_x, margin_y, rect_width, rect_height)
    
    # apply the grabcut algorithm with appropriate
    # values as parameters, number of iterations = 3 
    # cv2.GC_INIT_WITH_RECT is used because
    # of the rectangle mode is used 
    cv2.grabCut(image, mask, rectangle,  
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)
    
    # In the new mask image, pixels will 
    # be marked with four flags 
    # four flags denote the background / foreground 
    # mask is changed, all the 0 and 2 pixels 
    # are converted to the background
    # mask is changed, all the 1 and 3 pixels
    # are now the part of the foreground
    # the return type is also mentioned,
    # this gives us the final mask
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    
    # The final mask is multiplied with 
    # the input image to give the segmented image.
    image_segmented = image * mask2[:, :, np.newaxis]
    
    # Create binary mask for inpainting (flipped: black=inpaint, white=keep)
    # 0 where background, 255 where foreground
    binary_mask = np.where(mask2 == 0, 0, 255).astype('uint8')
    
    # Save both the segmented image and binary mask
    cv2.imwrite('output.png', image_segmented)
    cv2.imwrite('mask.png', binary_mask)
    print("Segmented image saved as output.png")
    print("Binary mask saved as mask.png")