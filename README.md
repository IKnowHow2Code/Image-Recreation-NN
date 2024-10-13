# Image-Recreation-NN

This repository contains a PyTorch-based neural network that uses a perceptual loss function to recreate images. The AI starts with a random noise image and iteratively optimizes it to resemble the target image. The entire process is visualized and saved as an MP4 video, allowing you to see how the AI gradually improves its reconstruction.

## Features
- **Perceptual Loss**: The network leverages a pre-trained VGG16 model to compare high-level image features rather than relying on pixel-level differences, which helps generate more realistic reconstructions.
- **High Resolution Support**: Recreate images at higher resolutions (e.g., 512x512) for better visual quality.
- **Video Output**: Save the progression of image recreation as an MP4 video, showing how the AI moves from noise to a refined image.
- **Customizable Steps**: Fine-tune the number of optimization steps to improve the quality of the final image or adjust processing time.

## Demo

Here's a preview of how the AI progressively recreates an image from noise:

![Sample Image Recreation](sample.gif)

## Requirements

- **Python 3.7+**
- **PyTorch 1.8+** with CUDA support for GPU acceleration (optional but recommended for faster performance)
- **torchvision**
- **matplotlib**
- **OpenCV** (for saving the video)

## Installation

You can either clone this repository using Git or download it as a ZIP file and extract it.

### Option 1: Clone the Repository

1. Open your terminal or command prompt and clone the repository:
   ```bash
   git clone https://github.com/IKnowHow2Code/Image-Recreation-NN.git
   cd Image-Recreation-NN
