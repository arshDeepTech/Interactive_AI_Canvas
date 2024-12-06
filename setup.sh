# Stable Diffusion Web App

A Flask-based web application for generating images using Stable Diffusion, optimized for systems with low GPU memory.

## Features

- Web interface for text-to-image generation
- Memory-optimized for low memory GPUs
- Automatic fallback to smaller resolutions if out of memory


## Requirements

- Python 3.8+
- CUDA-capable GPU with 4GB+ VRAM
- PyTorch with CUDA support
- Flask

## Installation

1. Clone the repository:

```bash
git clone https://github.com/arshDeepTech/Interactive_AI_Canvas.git
```

2. Install dependencies:

```bash
chmod +x setup.sh
bash setup.sh
```

3. Run the app:
    
```bash
python app.py
```

4. Open the browser and navigate to:

```bash
http://localhost:5000
```
