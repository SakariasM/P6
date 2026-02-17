import maskingService as masking
import inPaintingService as inpainting

def main():
    # Masking original input image
    input_image = 'image.png' 
    masking.mask_data(input_image)

    # Await masking, then inpaint the image with a specified image
    mask_image = 'mask.png'  # Use the binary mask, not the segmented image
    inpainting.inpaint_image(input_image, mask_image, "A pirate ship")

if __name__ == '__main__':
    main()



