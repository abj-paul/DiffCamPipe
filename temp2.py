import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# Filter Implementations (from previous artifact)
# --------------------------------

# 1. Sharpening Filter
class SharpeningFilter(nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        kernel_init = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        
        identity = torch.eye(3, dtype=torch.float32)
        kernel_init = identity + (kernel_init - identity) * strength
        
        kernel_init = kernel_init.unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(kernel_init)
    
    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)
        for i in range(c):
            channel = x[:, i:i+1, :, :]
            out[:, i:i+1, :, :] = F.conv2d(
                channel, 
                self.kernel.expand(1, 1, 3, 3),
                padding=1
            )
        return torch.clamp(out, 0, 1)

# 2. Edge Detection
class SobelEdgeDetection(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.sobel_x = nn.Parameter(sobel_x)
        self.sobel_y = nn.Parameter(sobel_y)
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.rgb_weights = nn.Parameter(torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32))
    
    def forward(self, x):
        b, c, h, w = x.shape
        gray = torch.sum(x * self.rgb_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        magnitude = magnitude / (torch.max(magnitude, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        edges = torch.sigmoid((magnitude - self.threshold) * 10)
        return edges.repeat(1, 3, 1, 1)

# 3. Median Filter
class DifferentiableMedianFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        out = torch.zeros_like(x)
        
        for channel in range(c):
            patches = F.unfold(x_padded[:, channel:channel+1, :, :], 
                              kernel_size=self.kernel_size, 
                              stride=1)
            patches = patches.reshape(b, self.kernel_size*self.kernel_size, h*w)
            sorted_patches, _ = torch.sort(patches, dim=1)
            median_idx = self.kernel_size * self.kernel_size // 2
            median_values = sorted_patches[:, median_idx, :]
            out[:, channel, :, :] = median_values.reshape(b, h, w)
        
        return out

# 4. Contrast Enhancement
class ContrastEnhancement(nn.Module):
    def __init__(self, alpha=1.5, beta=0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x):
        enhanced = self.alpha * x + self.beta
        return torch.clamp(enhanced, 0, 1)

# 5. Bilateral Filter
class BilateralFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma_space=1.0, sigma_color=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_space = nn.Parameter(torch.tensor(sigma_space))
        self.sigma_color = nn.Parameter(torch.tensor(sigma_color))
        self.padding = kernel_size // 2
        
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        spatial_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
        self.register_buffer('spatial_kernel', spatial_kernel)
    
    def forward(self, x):
        # For computational efficiency, we'll implement a simplified version
        # for integration in the pipeline
        b, c, h, w = x.shape
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        
        # Extract patches
        patches = F.unfold(x_padded, kernel_size=self.kernel_size, stride=1)
        patches = patches.view(b, c, self.kernel_size**2, h*w)
        
        # Get center pixels for each patch
        center_idx = self.kernel_size**2 // 2
        centers = patches[:, :, center_idx:center_idx+1, :]
        
        # Calculate color distance
        color_diff = patches - centers
        color_weight = torch.exp(-(color_diff**2) / (2 * self.sigma_color**2))
        
        # Apply spatial and color weights
        weight = self.spatial_kernel.view(1, 1, -1, 1) * color_weight
        weight = weight / (weight.sum(dim=2, keepdim=True) + 1e-8)
        
        # Apply weighted average
        out = torch.sum(patches * weight, dim=2)
        out = out.view(b, c, h, w)
        
        return out

# 6. Unsharp Masking
class UnsharpMasking(nn.Module):
    def __init__(self, strength=1.5, kernel_size=5, sigma=1.0):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    
    def forward(self, x):
        blurred = self.blur(x)
        mask = x - blurred
        sharpened = x + self.strength * mask
        return torch.clamp(sharpened, 0, 1)

# --------------------------------
# From your original code (reused)
# --------------------------------

# Gaussian Blur (from your original code)
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        channels = x.shape[1]
        kernel_1d = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        kernel_1d = torch.exp(-kernel_1d ** 2 / (2 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(channels, 1, -1, -1).to(x.device)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=channels)

# White Balance (from your original code)
class WhiteBalance(nn.Module):
    def __init__(self, init_gains=(1.2, 1.0, 0.9)):
        super().__init__()
        self.gains = nn.Parameter(torch.tensor(init_gains, dtype=torch.float32))

    def forward(self, x):
        return x * self.gains.view(1, -1, 1, 1)

# Gamma Correction (from your original code)
class GammaCorrection(nn.Module):
    def __init__(self, init_gamma=2.2):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, x):
        return torch.clamp(x, 1e-8, 1) ** (1 / self.gamma)

# Upsampling CNN (from your original code)
class UpsampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)

# --------------------------------
# Complete Pipeline Implementations
# --------------------------------

class CompletePipeline(nn.Module):
    """Complete differentiable image processing pipeline with filter, gamma correction, and upsampling"""
    def __init__(self, filter_type='bilateral'):
        super().__init__()
        # Downsampling (fixed operation, not learnable)
        self.downsample_factor = 0.5
        
        # Image processing components
        self.white_balance = WhiteBalance(init_gains=(1.2, 1.0, 0.9))
        
        # Choose filter based on filter_type
        if filter_type == 'sharpening':
            self.filter = SharpeningFilter()
        elif filter_type == 'edge_detection':
            self.filter = SobelEdgeDetection()
        elif filter_type == 'median':
            self.filter = DifferentiableMedianFilter()
        elif filter_type == 'contrast':
            self.filter = ContrastEnhancement()
        elif filter_type == 'bilateral':
            self.filter = BilateralFilter()
        elif filter_type == 'unsharp_masking':
            self.filter = UnsharpMasking()
        else:  # Default to Gaussian Blur
            self.filter = GaussianBlur()
        
        # Gamma correction
        self.gamma = GammaCorrection(init_gamma=2.2)
        
        # Upsampling CNN
        self.upsample = UpsampleCNN()
        
        self.filter_type = filter_type
    
    def forward(self, x_hr):
        # Downsample input (simulate low-res image)
        x_lr = F.interpolate(x_hr, scale_factor=self.downsample_factor, mode='bicubic', align_corners=False)
        
        # Apply white balance
        x_lr = self.white_balance(x_lr)
        
        # Apply specific filter
        x_lr = self.filter(x_lr)
        
        # Apply gamma correction
        x_lr = self.gamma(x_lr)
        
        # Apply upsampling CNN
        x_sr = self.upsample(x_lr)
        
        return x_sr, x_lr  # Return both the final output and the filtered low-res image

# --------------------------------
# Training and Evaluation
# --------------------------------

def train_complete_pipeline(model, train_loader, test_loader, num_epochs=75, lr=1e-3):
    """Train the complete image processing pipeline"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.6f}")
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_pipeline(model, test_loader, epoch + 1)
    
    # Save model
    torch.save(model.state_dict(), f"complete_pipeline_{model.filter_type}.pth")
    print(f"Model saved to complete_pipeline_{model.filter_type}.pth")
    
    return model

def evaluate_pipeline(model, test_loader, epoch=None, num_samples=5):
    """Evaluate the complete image processing pipeline"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs, filtered_lr = model(images)
            
            # Calculate metrics for each image in batch
            for i in range(images.size(0)):
                # Convert tensors to numpy arrays for metrics computation
                original_np = images[i].permute(1, 2, 0).cpu().numpy()
                output_np = outputs[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                
                # Calculate PSNR
                psnr_val = psnr(original_np, output_np, data_range=1.0)
                
                # Calculate SSIM
                ssim_val = ssim(original_np, output_np, data_range=1.0, 
                               multichannel=True, channel_axis=2)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_images += 1
                
                # Visualize a few samples
                if num_images <= num_samples:
                    # Downsampled image (before filter)
                    x_lr_before = F.interpolate(images[i].unsqueeze(0), scale_factor=0.5, 
                                             mode='bicubic', align_corners=False).squeeze(0)
                    
                    # Filtered LR image
                    x_lr_filtered = filtered_lr[i]
                    
                    visualize_pipeline_results(
                        images[i], 
                        x_lr_before,
                        x_lr_filtered,
                        outputs[i],
                        f"{model.filter_type}_epoch{epoch}" if epoch else model.filter_type,
                        psnr_val,
                        ssim_val
                    )
            
            # Process only a limited number of batches
            if num_images >= 100:
                break
    
    # Calculate averages
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    
    print(f"Filter: {model.filter_type}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def visualize_pipeline_results(original, downsampled, filtered_lr, output, filter_type, psnr_val, ssim_val):
    """Visualize original, downsampled, filtered, and final output images"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original HR image
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original HR Image")
    axes[0].axis('off')
    
    # Downsampled LR image (before filter)
    axes[1].imshow(downsampled.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Downsampled LR")
    axes[1].axis('off')
    
    # Filtered LR image
    axes[2].imshow(filtered_lr.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title(f"Filtered LR ({filter_type})")
    axes[2].axis('off')
    
    # Super-resolution output
    axes[3].imshow(output.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axes[3].set_title(f"Super-Resolution\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

# --------------------------------
# Main Execution
# --------------------------------

def main():
    # List of filter types to try
    filter_types = [
        'bilateral',       # Edge-preserving smoothing
        'sharpening',      # Edge enhancement
        'median',          # Noise reduction
        'contrast',        # Contrast enhancement
        'unsharp_masking'  # Another sharpening technique
    ]
    
    results = {}
    
    # Train and evaluate each filter type
    for filter_type in filter_types:
        print(f"\n===== Processing with {filter_type} filter =====")
        
        # Create the complete pipeline with the specified filter
        pipeline = CompletePipeline(filter_type=filter_type).to(device)
        
        # Train the pipeline
        pipeline = train_complete_pipeline(
            pipeline, 
            train_loader, 
            test_loader, 
            num_epochs=75,
            lr=1e-3
        )
        
        # Final evaluation
        psnr, ssim = evaluate_pipeline(pipeline, test_loader)
        results[filter_type] = {'psnr': psnr, 'ssim': ssim}
    
    # Print summary of results
    print("\n===== Summary of Results =====")
    print("-" * 50)
    for filter_type, metrics in results.items():
        print(f"{filter_type:15} - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()