# nerf_code

NeRF implementation 

# Frameworks

-PyTorch
-TensorFlow
-Jax


# Instalation

Create a virtual environment to isolate dependencies and activate it:

```
# On Windows:
python -m venv nerf_env
nerf_env\Scripts\activate 

# On macOS/Linux:
python3 -m venv nerf_env
source nerf_env/bin/activate  
```
# Note: If you are using a different CUDA version or a CPU-only setup, refer to the official PyTorch installation guide:
# https://pytorch.org/get-started/locally/

Install PyTorch based on your OS and CUDA version:

```
# On Windows:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# On macOS/Linux:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Test if you are installed corrcetly:

```
python
>>> import torch
>>> torch.cuda.is_available()

```
## Architecture

The NeRF (Neural Radiance Fields) method is a neural rendering technique that synthesizes novel views of complex 3D scenes by learning a continuous volumetric scene representation. Here's a high-level overview of its architecture:

### NeRF Architecture:

1. **Input Representation**:
   - NeRF takes as input a 3D spatial coordinate `(x, y, z)` and a viewing direction `(θ, φ)` (encoded as a 3D Cartesian vector).
   - These inputs are often encoded using a positional encoding technique (e.g., Fourier features) to capture high-frequency details.

2. **MLP (Multi-Layer Perceptron)**:
   - A fully connected neural network (MLP) maps the encoded inputs to outputs.
   - The MLP is split into two parts:
     - **First Part**: Maps the spatial coordinates `(x, y, z)` to a feature vector and a scalar density value (`σ`), which represents the volume density at that point.
     - **Second Part**: Combines the feature vector with the viewing direction `(θ, φ)` to predict the emitted RGB color `(r, g, b)`.

3. **Volume Rendering**:
   - NeRF uses volume rendering techniques to integrate the density (`σ`) and color (`r, g, b`) along rays cast through the scene.
   - This produces pixel values for the rendered image.

4. **Training**:
   - NeRF is trained using a set of input images and their corresponding camera poses.
   - The loss function minimizes the difference between the rendered pixel values and the ground truth pixel values from the input images.

5. **Positional Encoding**:
   - To handle high-frequency details, NeRF applies positional encoding to the input coordinates and viewing directions. This helps the MLP learn fine-grained scene details.


The current state of the art implementation of the NeRF are: 
- [nerfstudio][https://github.com/nerfstudio-project/nerfstudio]
- [instant-ngp][https://github.com/NVlabs/instant-ngp]