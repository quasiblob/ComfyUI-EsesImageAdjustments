# ==========================================================================
# Eses Image Adjustments V2
# ==========================================================================
#
# Description:
# The 'Eses Image Adjustments V2' node offers a suite of tools
# for image post-processing within ComfyUI. It allows users to fine-tune 
# various aspects of their images, applying effects in a sequential pipeline. 
# This version leverages PyTorch for all image processing, ensuring GPU 
# acceleration and efficient tensor operations.
#
# Key Features:
#
# - Global Tonal Adjustments:
#   - Contrast: Adjusts the difference between light and dark areas.
#   - Gamma: Controls mid-tone brightness.
#   - Saturation: Enhances or subdues image vibrancy.
#
# - Color Adjustments:
#   - Hue Rotation: Rotates the color spectrum of the image.
#   - RGB Channel Offsets: Allows for individual adjustments to the Red, Green,
#     and Blue color channels for precise color grading.
#
# - Creative Effects:
#   - Color Gel: Applies a colored tint to the image with adjustable strength.
#     The gel color can be specified using hex codes (e.g., #RRGGBB)
#     or RGB comma-separated values (e.g., R,G,B).
#
# - Sharpness:
#   - Sharpness: Adjusts the overall sharpness of the image.
#
# - Black & White Conversion:
#   - Grayscale: Easily converts the image to black and white.
#
# - Film Grain:
#   - Grain Strength: Controls the intensity of added film grain.
#   - Grain Contrast: Adjusts the contrast of the grain for more pronounced or subtle effects.
#   - Color Grain Mix: Blends between monochromatic and colored grain.
#
# Usage:
# Connect your image tensor to the 'image' input. Adjust parameters
# as needed. The node outputs the 'adjusted_image' tensor, maintaining
# compatibility with other ComfyUI nodes.
#
# Version: 1.0.0
# License: -
#
# ==========================================================================

import torch
import re
import torch.nn.functional as F

# --- Helper Functions for Image Adjustments (PyTorch-based) ---

# Helper to clamp values to [0, 1] for image tensors
def clamp_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, 0.0, 1.0)

def parse_color_string_to_tensor(color_str: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Parses a hex (#RRGGBB) or RGB comma-separated (R,G,B) string into an (R, G, B) tensor (0-1 range)."""
    color_str = color_str.strip()

    # Try hex format: #RRGGBB
    if re.match(r'^#[0-9a-fA-F]{6}$', color_str):
        try:
            hex_val = color_str[1:]
            r = int(hex_val[0:2], 16)
            g = int(hex_val[2:4], 16)
            b = int(hex_val[4:6], 16)
            return torch.tensor([r, g, b], device=device, dtype=dtype) / 255.0
        except ValueError:
            pass # Fall through to next attempt or default

    # Try RGB comma-separated format: R,G,B
    parts = color_str.split(',')
    if len(parts) == 3:
        try:
            r = float(parts[0].strip())
            g = float(parts[1].strip())
            b = float(parts[2].strip())
            # Clamp values to 0-255, then normalize to 0-1
            r = max(0.0, min(255.0, r)) / 255.0
            g = max(0.0, min(255.0, g)) / 255.0
            b = max(0.0, min(255.0, b)) / 255.0
            return torch.tensor([r, g, b], device=device, dtype=dtype)
        except ValueError:
            pass # Fall through to default

    # Default to white (1.0, 1.0, 1.0) if parsing fails
    print(f"Warning: Could not parse color string '{color_str}'. Defaulting to white (1.0,1.0,1.0) tensor.")
    return torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)

def adjust_contrast_tensor(image_tensor: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """Adjusts the contrast of a PyTorch Tensor (B, H, W, C)."""
    if contrast_factor == 1.0:
        return image_tensor
    # For contrast, compute the mean (gray point) and adjust deviations from it
    # Mean across H, W dimensions for each B, C
    mean = image_tensor.mean(dim=(-3, -2), keepdim=True) # (B, 1, 1, C)
    adjusted_tensor = mean + (image_tensor - mean) * contrast_factor
    return clamp_image_tensor(adjusted_tensor)

def adjust_gamma_tensor(image_tensor: torch.Tensor, gamma_factor: float) -> torch.Tensor:
    """Applies gamma correction to a PyTorch Tensor (B, H, W, C)."""
    if gamma_factor == 1.0:
        return image_tensor
    # Gamma correction formula: output = input^(1/gamma)
    adjusted_tensor = torch.pow(image_tensor, 1.0 / gamma_factor)
    return clamp_image_tensor(adjusted_tensor)

def adjust_saturation_tensor(image_tensor: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """Adjusts the saturation of a PyTorch Tensor (B, H, W, C)."""
    if saturation_factor == 1.0:
        return image_tensor

    # Convert to grayscale for desaturation baseline
    # Luminance formula (sRGB): 0.299*R + 0.587*G + 0.114*B
    grayscale = (image_tensor[..., 0] * 0.299 +
                 image_tensor[..., 1] * 0.587 +
                 image_tensor[..., 2] * 0.114).unsqueeze(-1) # (B, H, W, 1)

    # Blend original and grayscale based on saturation factor
    # Lerp: start * (1 - weight) + end * weight
    # Here, grayscale is 'start', original is 'end'
    # For desaturation, blend towards grayscale (weight < 1)
    # For oversaturation, blend beyond original (weight > 1)
    adjusted_tensor = torch.lerp(grayscale, image_tensor, saturation_factor)
    return clamp_image_tensor(adjusted_tensor)

def adjust_hue_rotation_tensor(image_tensor: torch.Tensor, hue_degrees: float) -> torch.Tensor:
    """Rotates the hue of a PyTorch Tensor (B, H, W, C)."""
    if hue_degrees == 0.0:
        return image_tensor

    # Convert RGB to HSV
    # This is a manual implementation of RGB to HSV and back.
    # It might be less optimized than dedicated libraries but adheres to constraints.
    max_rgb, _ = torch.max(image_tensor, dim=-1, keepdim=True)
    min_rgb, _ = torch.min(image_tensor, dim=-1, keepdim=True)
    delta = max_rgb - min_rgb

    v = max_rgb # Value is max(R, G, B)

    s = torch.where(max_rgb != 0, delta / max_rgb, torch.zeros_like(delta)) # Saturation

    hue = torch.zeros_like(image_tensor[..., :1]) # Initialize hue channel (B, H, W, 1)

    # Compute Hue
    # For R channel (max_rgb == image_tensor[..., 0])
    is_r_max = (max_rgb == image_tensor[..., 0:1]) # Use slice to keep dim
    hue_r_branch = torch.where(is_r_max & (delta != 0), (image_tensor[..., 1:2] - image_tensor[..., 2:3]) / delta, torch.zeros_like(hue))
    hue = torch.where(is_r_max, hue_r_branch, hue)

    # For G channel (max_rgb == image_tensor[..., 1])
    is_g_max = (max_rgb == image_tensor[..., 1:2])
    hue_g_branch = torch.where(is_g_max & (delta != 0), (image_tensor[..., 2:3] - image_tensor[..., 0:1]) / delta + 2, torch.zeros_like(hue))
    hue = torch.where(is_g_max, hue_g_branch, hue)

    # For B channel (max_rgb == image_tensor[..., 2])
    is_b_max = (max_rgb == image_tensor[..., 2:3])
    hue_b_branch = torch.where(is_b_max & (delta != 0), (image_tensor[..., 0:1] - image_tensor[..., 1:2]) / delta + 4, torch.zeros_like(hue))
    hue = torch.where(is_b_max, hue_b_branch, hue)

    hue = (hue * 60.0) % 360.0 # Convert to degrees and wrap around

    # Apply hue rotation
    hue_rotated = (hue + hue_degrees) % 360.0

    # Convert HSV back to RGB
    # Ensure all components are (B, H, W) before calculations
    h = (hue_rotated / 360.0).squeeze(-1) # H in [0,1]
    s = s.squeeze(-1) # S in [0,1]
    v = v.squeeze(-1) # V in [0,1]

    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    # Initialize R', G', B' components as (B, H, W)
    r_prime = torch.zeros_like(c)
    g_prime = torch.zeros_like(c)
    b_prime = torch.zeros_like(c)

    # Define masks for different hue segments
    mask_0_1_6 = (h >= 0) & (h < 1/6)
    mask_1_6_2_6 = (h >= 1/6) & (h < 2/6)
    mask_2_6_3_6 = (h >= 2/6) & (h < 3/6)
    mask_3_6_4_6 = (h >= 3/6) & (h < 4/6)
    mask_4_6_5_6 = (h >= 4/6) & (h < 5/6)
    mask_5_6_1 = (h >= 5/6) & (h <= 1.0 + 1e-6) # Use epsilon for float comparison to include 1.0

    r_prime[mask_0_1_6] = c[mask_0_1_6]
    g_prime[mask_0_1_6] = x[mask_0_1_6]

    r_prime[mask_1_6_2_6] = x[mask_1_6_2_6]
    g_prime[mask_1_6_2_6] = c[mask_1_6_2_6]

    g_prime[mask_2_6_3_6] = c[mask_2_6_3_6]
    b_prime[mask_2_6_3_6] = x[mask_2_6_3_6]

    g_prime[mask_3_6_4_6] = x[mask_3_6_4_6]
    b_prime[mask_3_6_4_6] = c[mask_3_6_4_6]

    r_prime[mask_4_6_5_6] = x[mask_4_6_5_6]
    b_prime[mask_4_6_5_6] = c[mask_4_6_5_6]

    r_prime[mask_5_6_1] = c[mask_5_6_1]
    b_prime[mask_5_6_1] = x[mask_5_6_1]

    # Unsqueeze back to (B, H, W, 1) before concatenating
    r_final = (r_prime + m).unsqueeze(-1)
    g_final = (g_prime + m).unsqueeze(-1)
    b_final = (b_prime + m).unsqueeze(-1)

    adjusted_tensor = torch.cat([r_final, g_final, b_final], dim=-1)

    return clamp_image_tensor(adjusted_tensor)

def adjust_color_balance_tensor(image_tensor: torch.Tensor, r_offset: float, g_offset: float, b_offset: float) -> torch.Tensor:
    """Adjusts the color balance by offsetting R, G, B channels of a PyTorch Tensor (B, H, W, C)."""
    if r_offset == 0.0 and g_offset == 0.0 and b_offset == 0.0:
        return image_tensor

    # Convert offsets from -100 to 100 scale to -1 to 1 scale for tensor addition
    r_offset_norm = r_offset / 255.0
    g_offset_norm = g_offset / 255.0
    b_offset_norm = b_offset / 255.0

    offsets = torch.tensor([r_offset_norm, g_offset_norm, b_offset_norm], device=image_tensor.device, dtype=image_tensor.dtype)
    # Add offsets, broadcasting across H, W, and B
    adjusted_tensor = image_tensor + offsets
    return clamp_image_tensor(adjusted_tensor)

def apply_color_gel_tensor(image_tensor: torch.Tensor, gel_color_str: str, gel_strength: float) -> torch.Tensor:
    """
    Applies a color gel effect to a PyTorch Tensor (B, H, W, C).
    """
    if gel_strength == 0.0:
        return image_tensor

    gel_color_tensor = parse_color_string_to_tensor(gel_color_str, image_tensor.device, image_tensor.dtype) # (3,)
    
    # Broadcast gel_color_tensor to match image_tensor shape for multiplication
    gel_color_broadcast = gel_color_tensor.view(1, 1, 1, 3) # (1, 1, 1, 3)

    # Multiply operation (like "multiply" blend mode)
    multiplied_tensor = image_tensor * gel_color_broadcast

    # Blend original and multiplied based on strength
    # lerp(start, end, weight) = start * (1 - weight) + end * weight
    adjusted_tensor = torch.lerp(image_tensor, multiplied_tensor, gel_strength)
    return clamp_image_tensor(adjusted_tensor)

def adjust_sharpness_tensor(image_tensor: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    """Adjusts the sharpness of a PyTorch Tensor using a simple convolution with replicate padding."""
    if sharpness_factor == 1.0:
        return image_tensor

    # Permute to (B, C, H, W) for conv2d
    img_permuted = image_tensor.permute(0, 3, 1, 2) # (B, C, H, W)

    # Standard sharpening kernel (sums to 1, retains brightness)
    sharpen_kernel = torch.tensor([[0., -1., 0.],
                                   [-1.,  5., -1.],
                                   [0., -1., 0.]], dtype=image_tensor.dtype, device=image_tensor.device)
    sharpen_kernel = sharpen_kernel.view(1, 1, 3, 3) # (out_channels, in_channels/groups, kH, kW)

    # Apply replicate padding before convolution to handle borders more naturally
    # For a 3x3 kernel, padding of 1 on each side is needed to maintain size
    padded_img_permuted = F.pad(img_permuted, (1, 1, 1, 1), mode='replicate') # (left, right, top, bottom)

    final_sharpened_channels = []
    for c in range(padded_img_permuted.shape[1]):
        # Apply convolution with padding=0 because we pre-padded
        sharpened_channel = F.conv2d(padded_img_permuted[:, c:c+1, :, :], sharpen_kernel, padding=0)
        final_sharpened_channels.append(sharpened_channel)

    final_sharpened_img_permuted = torch.cat(final_sharpened_channels, dim=1) # (B, C, H, W)

    # Blend original and the "sharpened version"
    # lerp(start, end, weight) = start * (1 - weight) + end * weight
    # If sharpness_factor=1, weight=0 => original.
    # If sharpness_factor=0, weight=-1.0 => original - (final_sharpened - original).
    # If sharpness_factor=2, weight=1.0 => final_sharpened.
    adjusted_tensor_permuted = torch.lerp(img_permuted, final_sharpened_img_permuted, sharpness_factor - 1.0)

    # Permute back to (B, H, W, C)
    adjusted_tensor = adjusted_tensor_permuted.permute(0, 2, 3, 1)
    return clamp_image_tensor(adjusted_tensor)

def convert_to_grayscale_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """Converts a PyTorch Tensor (B, H, W, C) to grayscale."""
    # Luminance formula: 0.299*R + 0.587*G + 0.114*B
    grayscale_tensor = (image_tensor[..., 0] * 0.299 +
                        image_tensor[..., 1] * 0.587 +
                        image_tensor[..., 2] * 0.114).unsqueeze(-1)
    # Replicate the single channel across RGB to make it a 3-channel grayscale image
    return grayscale_tensor.repeat(1, 1, 1, 3)

def add_film_grain_tensor(image_tensor: torch.Tensor, grain_strength: float, grain_contrast: float, color_grain_mix: float) -> torch.Tensor:
    """Adds realistic film grain (Gaussian noise) to a PyTorch Tensor (B, H, W, C)."""
    if grain_strength == 0.0:
        return image_tensor

    # Generate noise with same shape as image_tensor
    noise_shape = image_tensor.shape
    device = image_tensor.device
    dtype = image_tensor.dtype

    # Color noise (different for each channel) - scaled for 0-1 range
    color_noise = torch.randn(noise_shape, device=device, dtype=dtype) * grain_strength * 0.5

    # Grayscale noise (same across channels) - scaled for 0-1 range
    grayscale_noise_1ch = torch.randn(noise_shape[0], noise_shape[1], noise_shape[2], 1, device=device, dtype=dtype) * grain_strength * 0.5
    grayscale_noise = grayscale_noise_1ch.repeat(1, 1, 1, noise_shape[3])

    # Blend color and grayscale noise
    blended_noise = torch.lerp(grayscale_noise, color_noise, color_grain_mix)
    blended_noise *= grain_contrast # Apply contrast to the blended noise

    noisy_image_tensor = image_tensor + blended_noise
    return clamp_image_tensor(noisy_image_tensor)


# --- ComfyUI Node Class ---

class EsesImageAdjustments2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # (B, H, W, C) tensor
                
                # Global Tonal Adjustments
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

                # Color Adjustments
                "hue_rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "r_offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "g_offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "b_offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                
                # Creative Effects
                "gel_color": ("STRING", {"default": "255,200,0", "multiline": False}),
                "gel_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Sharpness
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),

                # B&W
                "grayscale": ("BOOLEAN", {"default": False}),

                # Grain
                "grain_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001}),
                "grain_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_grain_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("ADJUSTED_IMAGE", "ORIGINAL_MASK",)
    FUNCTION = "adjust_image"
    CATEGORY = "Eses Nodes/Image Adjustments" # Category remains the same, but the node name will be new

    def adjust_image(self, image: torch.Tensor, mask=None,
                     # Tonal Adjustments
                     contrast=1.0, gamma=1.0, saturation=1.0,
                     # Color Adjustments
                     hue_rotation=0.0,
                     r_offset=0.0, g_offset=0.0, b_offset=0.0,
                     # Creative Effects
                     gel_color="255,200,0", gel_strength=0.0,
                     # Sharpness
                     sharpness=1.0,
                     # B&W
                     grayscale=False,
                     # Grain
                     grain_strength=0.0, grain_contrast=1.0, color_grain_mix=1.0):

        # Input image is (B, H, W, C)
        current_image_tensor = image # Start with the input tensor

        # Base Tonal Adjustments
        current_image_tensor = adjust_contrast_tensor(current_image_tensor, contrast)
        current_image_tensor = adjust_gamma_tensor(current_image_tensor, gamma)
        current_image_tensor = adjust_saturation_tensor(current_image_tensor, saturation)
        
        # Color Adjustments
        current_image_tensor = adjust_hue_rotation_tensor(current_image_tensor, hue_rotation)
        current_image_tensor = adjust_color_balance_tensor(current_image_tensor, r_offset, g_offset, b_offset)
        
        # Creative Effects
        current_image_tensor = apply_color_gel_tensor(current_image_tensor, gel_color, gel_strength)

        # Sharpness
        current_image_tensor = adjust_sharpness_tensor(current_image_tensor, sharpness)
        
        # Grayscale
        if grayscale:
            current_image_tensor = convert_to_grayscale_tensor(current_image_tensor)
        
        # Grain
        current_image_tensor = add_film_grain_tensor(current_image_tensor, grain_strength, grain_contrast, color_grain_mix)

        # The final output is already a batch tensor
        adjusted_image_tensor = current_image_tensor

        return (adjusted_image_tensor, mask,)
