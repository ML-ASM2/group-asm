"""
Image‑handling utilities for the Paddy demo.

Functions
---------
load_image(uploaded_file) -> (PIL.Image, np.ndarray)
    Opens an uploaded JPG/PNG and returns both the PIL image and a NumPy copy.
to_tensor(pil_img) -> torch.Tensor
    Resizes → normalises → converts to 1×C×H×W float32 tensor for CNN/MLP.
"""

from PIL import Image
import numpy as np
import torch

# ---------------------------------------------------------------------
IMG_SIZE = 224          # resize target; adjust to your CNN's input size
# ---------------------------------------------------------------------


def load_image(uploaded_file):
    """
    Parameters
    ----------
    uploaded_file : UploadedFile
        object returned by st.file_uploader

    Returns
    -------
    pil_img : PIL.Image.Image
    arr      : np.ndarray
    """
    pil_img = Image.open(uploaded_file).convert("RGB")
    return pil_img, np.array(pil_img)


def to_tensor(pil_img):
    """
    Rescale and format a PIL image for PyTorch inference.

    Returns
    -------
    torch.Tensor  (shape = 1×3×IMG_SIZE×IMG_SIZE, dtype = float32)
    """
    img_r = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_r).astype("float32") / 255.0          # 0‑1 range
    arr = arr.transpose(2, 0, 1)                             # C, H, W
    return torch.tensor(arr).unsqueeze(0)                    # add batch dim
