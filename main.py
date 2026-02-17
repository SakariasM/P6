import argparse
import maskingService as masking
import inPaintingService as inpainting

def main():
    parser = argparse.ArgumentParser(description='AI-powered image inpainting tool')
    parser.add_argument('prompt', type=str, help='Text prompt for AI to generate the new background')
    
    args = parser.parse_args()
    
    # Masking original input image
    input_image = 'image.png'
    masking.mask_data(input_image)

    # Await masking, then inpaint the image with a specified image
    mask_image = 'mask.png'  # Use the binary mask, not the segmented image
    inpainting.inpaint_image(input_image, mask_image, args.prompt)

if __name__ == '__main__':
    main()



