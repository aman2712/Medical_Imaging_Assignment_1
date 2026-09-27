import pydicom

import numpy as np
import torch
import torch.nn.functional as F

TARGET_SIZE = (224, 224) 

def dicom_to_tensor(path, target_size=TARGET_SIZE):
    ds = pydicom.dcmread(path)

    pixels = ds.pixel_array.astype(np.float32)

    # correct images where lower pixel values represent brighter areas
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # apply dicom intensity conversion if available
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    # normalize to [0, 1]
    lower, upper = np.percentile(pixels, [1, 99])
    pixels = np.clip(pixels, lower, upper)
    pixels = (pixels - lower) / (upper - lower + 1e-8)

    # convert from numpy format to a one-channel PyTorch tensor
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

    # resize to target size
    tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    return tensor.squeeze(0)