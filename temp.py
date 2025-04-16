import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# 1. Sharpening Filter
# --------------------------------

# Traditional implementation
def apply_sharpening(img_tensor, strength=1.0):
    """Apply sharpening filter using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Create sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Adjust strength
    kernel = np.identity(3, dtype=np.float32) + (kernel - np.identity(3, dtype=np.float32)) * strength
    
    # Apply filter
    sharpened = cv2.filter2D(img_np, -1, kernel)
    
    # Convert back to tensor
    img_tensor_sharpened = torch.from_numpy(sharpened.astype(np.float32) / 255.0).permute(2, 0, 1)
    return torch.clamp(img_tensor_sharpened, 0, 1)

# Differentiable implementation
class SharpeningFilter(nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        # Initialize learnable kernel with traditional sharpening kernel
        kernel_init = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        
        # Adjust strength
        identity = torch.eye(3, dtype=torch.float32)
        kernel_init = identity + (kernel_init - identity) * strength
        
        # Convert to format expected by conv2d (out_channels, in_channels, height, width)
        kernel_init = kernel_init.unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(kernel_init)
        self.strength = nn.Parameter(torch.tensor(strength))
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Apply convolution per channel
        out = torch.zeros_like(x)
        for i in range(c):
            channel = x[:, i:i+1, :, :]
            out[:, i:i+1, :, :] = F.conv2d(
                channel, 
                self.kernel.expand(1, 1, 3, 3),
                padding=1
            )
        
        return torch.clamp(out, 0, 1)

# --------------------------------
# 2. Edge Detection (Sobel)
# --------------------------------

# Traditional implementation
def apply_sobel_edge_detection(img_tensor, threshold=0.1):
    """Apply Sobel edge detection using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize
    magnitude = magnitude / magnitude.max()
    
    # Apply threshold
    edges = (magnitude > threshold).astype(np.float32)
    
    # Convert to 3-channel output
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    
    # Convert back to tensor
    img_tensor_edges = torch.from_numpy(edges_3ch).permute(2, 0, 1)
    return img_tensor_edges

# Differentiable implementation
class SobelEdgeDetection(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        
        # Initialize Sobel kernels
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
        
        # RGB to grayscale conversion weights
        self.rgb_weights = nn.Parameter(torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32))
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Convert to grayscale
        gray = torch.sum(x * self.rgb_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        
        # Apply Sobel filters
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Normalize
        magnitude = magnitude / (torch.max(magnitude, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        
        # Apply threshold using sigmoid for smooth differentiable threshold
        edges = torch.sigmoid((magnitude - self.threshold) * 10)
        
        # Repeat across channels
        edges = edges.repeat(1, 3, 1, 1)
        
        return edges

# --------------------------------
# 3. Median Filter (Noise Reduction)
# --------------------------------

# Traditional implementation
def apply_median_filter(img_tensor, kernel_size=3):
    """Apply median filter using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Apply median filter
    filtered = cv2.medianBlur(img_np, kernel_size)
    
    # Convert back to tensor
    img_tensor_filtered = torch.from_numpy(filtered.astype(np.float32) / 255.0).permute(2, 0, 1)
    return img_tensor_filtered

# Differentiable approximation of median filter
class DifferentiableMedianFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        
        # Parameter to control the "softness" of the median approximation
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Pad the input
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        
        # Storage for output
        out = torch.zeros_like(x)
        
        # Process each channel separately
        for channel in range(c):
            # Extract patches
            patches = F.unfold(x_padded[:, channel:channel+1, :, :], 
                              kernel_size=self.kernel_size, 
                              stride=1)
            
            # Reshape patches to (batch, kernel_size*kernel_size, h*w)
            patches = patches.reshape(b, self.kernel_size*self.kernel_size, h*w)
            
            # Sort patches along the kernel dimension
            sorted_patches, _ = torch.sort(patches, dim=1)
            
            # Get median value (middle of the sorted values)
            median_idx = self.kernel_size * self.kernel_size // 2
            median_values = sorted_patches[:, median_idx, :]
            
            # Reshape back to image shape
            out[:, channel, :, :] = median_values.reshape(b, h, w)
        
        return out

# --------------------------------
# 4. Contrast Enhancement
# --------------------------------

# Traditional implementation
def apply_contrast_enhancement(img_tensor, alpha=1.5, beta=0):
    """Apply contrast enhancement using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Apply contrast enhancement
    enhanced = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    # Convert back to tensor
    img_tensor_enhanced = torch.from_numpy(enhanced.astype(np.float32) / 255.0).permute(2, 0, 1)
    return torch.clamp(img_tensor_enhanced, 0, 1)

# Differentiable implementation
class ContrastEnhancement(nn.Module):
    def __init__(self, alpha=1.5, beta=0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x):
        # Apply contrast enhancement
        enhanced = self.alpha * x + self.beta
        return torch.clamp(enhanced, 0, 1)

# --------------------------------
# 5. JPEG Compression Artifacts Removal
# --------------------------------

# Traditional implementation
def apply_jpeg_artifacts_removal(img_tensor, h=10, templateWindowSize=7, searchWindowSize=21):
    """Apply JPEG artifacts removal using traditional method (uses Non-local Means Denoising)"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Apply Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_np, 
        None, 
        h=h,
        hColor=h,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )
    
    # Convert back to tensor
    img_tensor_denoised = torch.from_numpy(denoised.astype(np.float32) / 255.0).permute(2, 0, 1)
    return img_tensor_denoised

# Differentiable approximation of JPEG artifacts removal
class JPEGArtifactsRemoval(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        # Use a small U-Net-like architecture
        # Encoder
        self.enc1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Decoder
        self.dec3 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1)
        self.dec1 = nn.Conv2d(32, channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        
        # Decoder
        d3 = self.relu(self.dec3(e3))
        d2 = self.relu(self.dec2(d3))
        d1 = self.dec1(d2)
        
        # Skip connection from input
        out = x + d1
        
        return torch.clamp(out, 0, 1)

# --------------------------------
# 6. Bilateral Filter
# --------------------------------

# Traditional implementation
def apply_bilateral_filter(img_tensor, d=9, sigmaColor=75, sigmaSpace=75):
    """Apply bilateral filter using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(img_np, d, sigmaColor, sigmaSpace)
    
    # Convert back to tensor
    img_tensor_filtered = torch.from_numpy(filtered.astype(np.float32) / 255.0).permute(2, 0, 1)
    return img_tensor_filtered

# Differentiable approximation of bilateral filter
class BilateralFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma_space=1.0, sigma_color=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_space = nn.Parameter(torch.tensor(sigma_space))
        self.sigma_color = nn.Parameter(torch.tensor(sigma_color))
        self.padding = kernel_size // 2
        
        # Create spatial kernel
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)
        spatial_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
        self.register_buffer('spatial_kernel', spatial_kernel)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Create output tensor
        out = torch.zeros_like(x)
        
        # Process each batch and channel separately
        for batch_idx in range(b):
            for channel_idx in range(c):
                img = x[batch_idx, channel_idx]
                
                # Pad the image
                padded = F.pad(img.unsqueeze(0).unsqueeze(0), 
                              (self.padding, self.padding, self.padding, self.padding), 
                              mode='reflect').squeeze()
                
                # Create output for this channel
                out_channel = torch.zeros_like(img)
                
                # Iterate over each pixel in the output
                for i in range(h):
                    for j in range(w):
                        # Extract patch around current pixel
                        patch = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                        
                        # Current pixel value
                        center_val = img[i, j]
                        
                        # Compute color weight
                        color_diff = patch - center_val
                        color_weight = torch.exp(-(color_diff**2) / (2 * self.sigma_color**2))
                        
                        # Apply spatial and color weights
                        weight = self.spatial_kernel * color_weight
                        weight = weight / (torch.sum(weight) + 1e-8)
                        
                        # Compute weighted average
                        out_channel[i, j] = torch.sum(weight * patch)
                
                out[batch_idx, channel_idx] = out_channel
        
        return out

# --------------------------------
# 7. Unsharp Masking
# --------------------------------

# Traditional implementation
def apply_unsharp_masking(img_tensor, strength=1.5, radius=5, threshold=0):
    """Apply unsharp masking using traditional method"""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Blur the image
    blurred = cv2.GaussianBlur(img_np, (radius, radius), 0)
    
    # Create the unsharp mask
    mask = cv2.subtract(img_np, blurred)
    
    # Apply the mask to the original image with the specified strength
    sharpened = cv2.addWeighted(img_np, 1.0, mask, strength, 0)
    
    # Convert back to tensor
    img_tensor_sharpened = torch.from_numpy(sharpened.astype(np.float32) / 255.0).permute(2, 0, 1)
    return torch.clamp(img_tensor_sharpened, 0, 1)

# Differentiable implementation
class UnsharpMasking(nn.Module):
    def __init__(self, strength=1.5, kernel_size=5, sigma=1.0):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    
    def forward(self, x):
        # Blur the image
        blurred = self.blur(x)
        
        # Create the mask
        mask = x - blurred
        
        # Apply the mask with learnable strength
        sharpened = x + self.strength * mask
        
        return torch.clamp(sharpened, 0, 1)

# --------------------------------
# Full Pipeline Combining All Filters
# --------------------------------

class TraditionalFilterPipeline:
    """Pipeline for applying traditional filters sequentially"""
    def __init__(self):
        pass
    
    def __call__(self, x, filter_type):
        if filter_type == 'sharpening':
            return apply_sharpening(x)
        elif filter_type == 'edge_detection':
            return apply_sobel_edge_detection(x)
        elif filter_type == 'median':
            return apply_median_filter(x)
        elif filter_type == 'contrast':
            return apply_contrast_enhancement(x)
        elif filter_type == 'jpeg_artifacts':
            return apply_jpeg_artifacts_removal(x)
        elif filter_type == 'bilateral':
            return apply_bilateral_filter(x)
        elif filter_type == 'unsharp_masking':
            return apply_unsharp_masking(x)
        else:
            return x

class DifferentiableFilterPipeline(nn.Module):
    """Pipeline for applying differentiable filters sequentially"""
    def __init__(self):
        super().__init__()
        self.sharpening = SharpeningFilter()
        self.edge_detection = SobelEdgeDetection()
        self.median = DifferentiableMedianFilter()
        self.contrast = ContrastEnhancement()
        self.jpeg_artifacts = JPEGArtifactsRemoval()
        self.bilateral = BilateralFilter()
        self.unsharp_masking = UnsharpMasking()
    
    def forward(self, x, filter_type):
        if filter_type == 'sharpening':
            return self.sharpening(x)
        elif filter_type == 'edge_detection':
            return self.edge_detection(x)
        elif filter_type == 'median':
            return self.median(x)
        elif filter_type == 'contrast':
            return self.contrast(x)
        elif filter_type == 'jpeg_artifacts':
            return self.jpeg_artifacts(x)
        elif filter_type == 'bilateral':
            return self.bilateral(x)
        elif filter_type == 'unsharp_masking':
            return self.unsharp_masking(x)
        else:
            return x

# --------------------------------
# Training and Evaluation Helper Functions
# --------------------------------

def train_filter_model(model, train_loader, filter_type, num_epochs=10, lr=1e-3):
    """Train a differentiable filter model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    traditional_pipeline = TraditionalFilterPipeline()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Create target outputs using traditional methods
            targets = []
            for i in range(batch_size):
                target = traditional_pipeline(images[i], filter_type)
                targets.append(target)
            targets = torch.stack(targets).to(device)
            
            # Forward pass
            outputs = model(images, filter_type)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    return model

def evaluate_filter_model(model, test_loader, filter_type, num_samples=10):
    """Evaluate differentiable filter model against traditional implementation"""
    model.eval()
    traditional_pipeline = TraditionalFilterPipeline()
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # Apply traditional filter
                traditional_output = traditional_pipeline(images[i], filter_type)
                
                # Apply differentiable filter
                differentiable_output = model(images[i].unsqueeze(0), filter_type).squeeze(0)
                
                # Calculate metrics
                psnr_val = calculate_psnr(traditional_output, differentiable_output)
                ssim_val = calculate_ssim(traditional_output, differentiable_output)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1
                
                # Visualize sample results
                if count <= num_samples:
                    visualize_results(images[i], traditional_output, differentiable_output, filter_type)
                
                if count >= 100:  # Limit evaluation to 100 images
                    break
            
            if count >= 100:
                break
    
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print(f"Filter: {filter_type}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(img1, img2):
    """Simple SSIM calculation between two images"""
    # Implementation simplified for demonstration
    # In practice, use the skimage or pytorch-msssim implementations
    c1 = (0.01 * 1) ** 2
    c2 = (0.03 * 1) ** 2
    
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return torch.mean(ssim_map)

def visualize_results(original, traditional, differentiable, filter_type):
    """Visualize original image, traditional filter result, and differentiable filter result"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Traditional filter result
    axes[1].imshow(traditional.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f"Traditional {filter_type}")
    axes[1].axis('off')
    
    # Differentiable filter result
    axes[2].imshow(differentiable.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title(f"Differentiable {filter_type}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# --------------------------------
# Main Execution
# --------------------------------

def main():
    # Create dataset and dataloaders
    # (This assumes you have the same dataset loading code as in your original script)
    
    # Create differentiable filter pipeline
    differentiable_pipeline = DifferentiableFilterPipeline().to(device)
    
    # List of filter types to train and evaluate
    filter_types = [
        'sharpening',
        'edge_detection',
        'median',
        'contrast',
        'jpeg_artifacts',
        'bilateral',
        'unsharp_masking'
    ]
    
    # Train and evaluate each filter
    results = {}
    for filter_type in filter_types:
        print(f"\nTraining {filter_type} filter...")
        train_filter_model(differentiable_pipeline, train_loader, filter_type, num_epochs=5)
        
        print(f"\nEvaluating {filter_type} filter...")
        psnr, ssim = evaluate_filter_model(differentiable_pipeline, test_loader, filter_type)
        results[filter_type] = {'psnr': psnr, 'ssim': ssim}
    
    # Save model
    torch.save(differentiable_pipeline.state_dict(), "differentiable_filters.pth")
    
    # Print overall results
    print("\nSummary of Results:")
    print("-" * 50)
    for filter_type, metrics in results.items():
        print(f"{filter_type:20} - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()