# P6

## Prerequisites

- Python 3.12 or compatible version
- ~5GB free disk space (for AI model download on first run)

## Setup Instructions

1. **Clone the repository**

2. **Create a virtual environment** (recommended but optional)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   
   *Skip this step if you prefer to install packages globally, though this may cause conflicts with other Python projects.*

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This will install numpy, opencv-python, matplotlib, torch, diffusers, transformers, pillow, and accelerate.

4. **Add your input image**
   - Place your image in the project directory as `image.png`
   - The image should contain an object you want to keep with a background you want to replace

5. **Run the program**
   ```bash
   python main.py "your prompt here"
   ```
   
   Example:
   ```bash
   python main.py "A pirate ship sailing on the ocean"
   ```

## Usage

### Basic Usage

Run the program with a text prompt describing what you want the background to be:

```bash
python main.py "A pirate ship"
```

Example prompts:
```bash
python main.py "A snowy mountain landscape"
python main.py "A tropical beach at sunset"
python main.py "A futuristic city skyline"
```

### Get Help

To see all available options:

```bash
python main.py --help
```

### Output Files

The program generates three files:
- `output.png` - Segmented foreground object
- `mask.png` - Binary mask used for inpainting
- `inpainted_output.png` - Final result with AI-generated background

### First Run

The first time you run the program, it will download the Stable Diffusion inpainting model (~5GB). This may take several minutes depending on your internet connection.

## Performance Notes

- The program runs on CPU by default
- Image generation takes several minutes on CPU (5-10 minutes depending on your hardware)
- For faster performance, consider running on a system with CUDA-compatible GPU

## How It Works

1. **Masking Service**: Uses OpenCV's GrabCut algorithm to automatically segment the foreground object
2. **Inpainting Service**: Uses Stable Diffusion to generate new background content based on your text prompt
3. The foreground object is preserved while the background is replaced with AI-generated imagery

## Troubleshooting

**Error: `cannot import name 'MT5Tokenizer'`**
- Run: `pip install "diffusers==0.30.0" "transformers==4.45.0"`

**Program seems stuck at 100%**
- This is normal! The final processing takes time on CPU. Wait a few more minutes.

**Out of memory errors**
- Close other applications to free up RAM
- The model requires ~8GB RAM minimum
