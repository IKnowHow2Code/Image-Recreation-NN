# Image-Recreation-NN

This repository contains a PyTorch-based neural network for recreating images using a perceptual loss function. The AI starts with a random noise image and iteratively optimizes it to resemble the target image. The process is visualized by saving the progression as an MP4 video.

## Features
- **Perceptual Loss**: Uses a pre-trained VGG16 model to compare high-level features of images.
- **High Resolution**: Supports recreating images at higher resolutions (e.g., 512x512).
- **Video Output**: Tracks the progression of the AI's attempts to recreate the image and saves it as an MP4 video.
- **Customizable Steps**: You can set the number of optimization steps for fine-tuning the image recreation process.

## Demo

![Sample Image Recreation](sample.gif)

## Requirements

- Python 3.7+
- PyTorch 1.8+ with CUDA support (for GPU acceleration)
- torchvision
- matplotlib
- OpenCV (for saving video)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Image-Recreation-NN.git
   cd Image-Recreation-NN
