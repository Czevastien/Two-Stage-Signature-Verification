import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class ResizePad:
    """Resize image keeping aspect ratio, then pad to square (180x180)."""

    def __init__(self, size=180, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, fill=self.fill)

        return img
