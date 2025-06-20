# Eses Image Adjustments

![Eses Image Adjustments Node Screenshot](docs/image_adjustments.png)

## Description

The 'Eses Image Adjustments' is a ComfyUI custom node designed for simple and easy to use image post-processing. It provides a sequential pipeline for fine-tuning various image aspects, utilizing PyTorch for GPU acceleration and efficient tensor operations. If you want a node with minimal dependecies, and don't want to download ten nodes to get images adjustment features, then give this one a try! 

This is not a replacement for separate image adjustment nodes, as you can't reorder operations here (these are done pretty much in order you see the UI elements). This node is meant for quick and easy to use color adjustments. Film grain is relatively fast (mainly the reason I put this together), 4000x6000 pixels image takes ~2-3 seconds to process 'on my machine' (lol).

## Features

* **Global Tonal Adjustments**:
    * **Contrast**: Modifies the distinction between light and dark areas.
    * **Gamma**: Manages mid-tone brightness.
    * **Saturation**: Controls the vibrancy of image colors.
* **Color Adjustments**:
    * **Hue Rotation**: Rotates the entire color spectrum of the image.
    * **RGB Channel Offsets**: Enables precise color grading through individual adjustments to Red, Green, and Blue channels.
* **Creative Effects**:
    * **Color Gel**: Applies a customizable colored tint to the image. The gel color can be specified using hex codes (e.g., `#RRGGBB`) or RGB comma-separated values (e.g., `R,G,B`). Adjustable strength controls the intensity of the tint.
* **Sharpness**:
    * **Sharpness**: Adjusts the overall sharpness of the image.
* **Black & White Conversion**:
    * **Grayscale**: Converts the image to black and white with a single toggle.
* **Film Grain**:
    * **Grain Strength**: Controls the intensity of the added film grain.
    * **Grain Contrast**: Adjusts the contrast of the grain for either subtle or pronounced effects.
    * **Color Grain Mix**: Blends between monochromatic and colored grain.


## Requirements

* Python >= 3.9  (tested only with Python 3.12)
* PyTorch – for image processing operations (you should have this if you have ComfyUI installed).
* Ensure you have ComfyUI installed and properly configured before adding custom nodes.


## Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/quasiblob/ComfyUI-EsesImageAdjustments.git
    ```
2.  **Place the custom node files in your ComfyUI installation:**
    Typically, custom nodes reside in the `ComfyUI/custom_nodes/` directory. Place the entire `ComfyUI-EsesImageAdjustments` folder into this directory. Consult the [ComfyUI documentation](https://github.com/comfyanonymous/ComfyUI/blob/master/README.md) for detailed instructions on custom node installation.

3.  **Install dependencies:**
    Navigate to the `ComfyUI-EsesImageAdjustments` directory and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Folder Structure

```
ComfyUI-EsesImageAdjustments/
├── init.py                 # Main module defining the custom nodes.
├── image_adjustsments_2.py # The Python file containing the node logic.
├── README.md               # This file.
├── requirements.txt        # Python package dependencies.
└── docs/                   # Optional documentation assets (images, etc.)
```


## Usage

Connect your image tensor to the 'image' input of the `Eses Image Adjustments 2` node within ComfyUI. Adjust the desired parameters using the provided sliders and input fields. The node outputs the `adjusted_image` tensor, maintaining compatibility with other ComfyUI nodes. An optional `mask` input is also available, which will be passed through to the `ORIGINAL_MASK` output.

## Inputs

* **image** (`IMAGE`): The input image tensor in `(B, H, W, C)` format.

* **contrast** (`FLOAT`): Adjusts image contrast.
    * Default: `1.0`
    * Min: `0.0`
    * Max: `2.0`
    * Step: `0.01`
<br><br>

* **gamma** (`FLOAT`): Applies gamma correction.
    * Default: `1.0`
    * Min: `0.1`
    * Max: `5.0`
    * Step: `0.01`
<br><br>

* **saturation** (`FLOAT`): Adjusts image saturation.
    * Default: `1.0`
    * Min: `0.0`
    * Max: `2.0`
    * Step: `0.01`
<br><br>

* **hue\_rotation** (`FLOAT`): Rotates the hue in degrees.
    * Default: `0.0`
    * Min: `-180.0`
    * Max: `180.0`
    * Step: `1.0`
<br><br>

* **r\_offset** (`FLOAT`): Red channel offset.
    * Default: `0.0`
    * Min: `-100.0`
    * Max: `100.0`
    * Step: `1.0`
<br><br>

* **g\_offset** (`FLOAT`): Green channel offset.
    * Default: `0.0`
    * Min: `-100.0`
    * Max: `100.0`
    * Step: `1.0`
<br><br>

* **b\_offset** (`FLOAT`): Blue channel offset.
    * Default: `0.0`
    * Min: `-100.0`
    * Max: `100.0`
    * Step: `1.0`
<br><br>

* **gel\_color** (`STRING`): The color for the gel effect (e.g., `"255,200,0"` or `"#FFC800"`).
    * Default: `"255,200,0"`
* **gel\_strength** (`FLOAT`): Strength of the color gel effect.
    * Default: `0.0`
    * Min: `0.0`
    * Max: `1.0`
    * Step: `0.01`
<br><br>

* **sharpness** (`FLOAT`): Adjusts image sharpness.
    * Default: `1.0`
    * Min: `0.0`
    * Max: `2.0`
    * Step: `0.01`
<br><br>

* **grayscale** (`BOOLEAN`): Converts the image to grayscale if `True`.
    * Default: `False`
<br><br>

* **grain\_strength** (`FLOAT`): Intensity of the film grain.
    * Default: `0.0`
    * Min: `0.0`
    * Max: `0.1`
    * Step: `0.001`
<br><br>

* **grain\_contrast** (`FLOAT`): Contrast of the film grain.
    * Default: `1.0`
    * Min: `0.0`
    * Max: `2.0`
    * Step: `0.01`
<br><br>

* **color\_grain\_mix** (`FLOAT`): Blend between monochromatic and colored grain.
    * Default: `1.0`
    * Min: `0.0`
    * Max: `1.0`
    * Step: `0.01`
<br><br>

* **mask** (`MASK`, *optional*): An optional mask tensor that will be passed through to the output.

## Outputs

* **ADJUSTED\_IMAGE** (`IMAGE`): The processed image tensor.
* **ORIGINAL\_MASK** (`MASK`): The mask tensor passed through from the input, if provided.

## Category

Eses Nodes/Image Adjustments

## Version

1.0.0

## License

- (Not yet specified)

## Contributing

- Feel free to report bugs and improvement ideas in issues, but I may not have time to do anything.

## Acknowledgements

Thanks to the ComfyUI community for their ongoing work.
