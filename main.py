import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Load a pre-trained model (like VGG) for perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg)[:16]).eval()  # Use first few layers of VGG
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze VGG weights during optimization
    
    def forward(self, img1, img2):
        return torch.nn.functional.mse_loss(self.features(img1), self.features(img2))

# Function to load and transform input images to the format needed for the network
def image_loader(image_name, device, high_res=True):
    # If high_res is True, we use a higher resolution than 224x224, such as 512x512
    size = (512, 512) if high_res else (224, 224)
    loader = T.Compose([T.Resize(size), T.ToTensor()])
    image = Image.open(image_name).convert('RGB')  # Convert to RGB to remove alpha channel
    image = loader(image).unsqueeze(0).to(device)  # Add batch dimension and move to the right device
    return image

# Function to convert tensor image back to a format that can be saved or displayed
def tensor_to_image(tensor):
    image = tensor.cpu().clone()  # Clone the tensor to avoid modifying the original
    image = image.squeeze(0)  # Remove batch dimension
    image = T.ToPILImage()(image)  # Convert tensor back to PIL image
    return np.array(image)  # Convert PIL image to numpy array (for video saving)

# AI attempts to recreate the image by generating a new one and optimizing it
def recreate_image(target_image, device, num_steps=1000, save_video=True):
    generated_image = torch.rand_like(target_image, device=device, requires_grad=True)
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam([generated_image], lr=0.01)

    frames = []  # To store the images for video creation

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = perceptual_loss(generated_image, target_image)  # Calculate loss
        loss.backward()
        optimizer.step()

        # Save frames for the video every 10 steps
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            current_image = tensor_to_image(generated_image)
            frames.append(current_image)
        
    if save_video:
        save_as_video(frames)  # Save the progression as an mp4 video

    return generated_image

# Save the progression as an MP4 video
def save_as_video(frames, output_file="output_video.mp4", fps=20):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    video.release()

# Main process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "example.png"  # Provide the path to the image you want to use

# Load the target image in high resolution
target_image = image_loader(image_path, device, high_res=True)

# AI recreates the image over many steps, while saving the video
generated_image = recreate_image(target_image, device, num_steps=1000, save_video=True)

# Display the final recreated image
plt.imshow(tensor_to_image(generated_image))
plt.title("Final Recreated Image")
plt.show()
