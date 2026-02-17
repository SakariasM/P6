import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

def inpaint_image(input_image, mask_image, prompt):
    print("Starting inpainting...")
    print(f"Loading model (this may take a minute on first run)...")
    # Using SD 1.5 inpainting - faster and lighter than Kandinsky
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float32  # float32 is better for CPU
    )
    print("Model loaded!")
    
    print("Loading images...")
    init_image = load_image(input_image)
    masked_image = load_image(mask_image)

    print("Generating image with AI model (this will take several minutes on CPU)...")
    generator = torch.Generator("cpu").manual_seed(92)
    image = pipeline(
        prompt=prompt, 
        image=init_image, 
        mask_image=masked_image, 
        generator=generator,
        num_inference_steps=20  # Reduced steps for faster generation
    ).images[0]
    
    print("Image generated, saving...")
    # Save the generated image
    image.save('inpainted_output.png')
    print("✓ Successfully saved inpainted_output.png")
    
    return image