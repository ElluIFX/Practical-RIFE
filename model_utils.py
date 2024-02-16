from loguru import logger
import numpy as np
import torch
from torch.nn import functional as F


def make_inference(model, I0, I1, n, scale):
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) / (n + 1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(model, I0, middle, n=n // 2)  # type: ignore
        second_half = make_inference(model, middle, I1, n=n // 2)  # type: ignore
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def calc_padding(w, h, scale):
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    return padding


def pad_image(img, padding, fp16=False):
    if fp16:
        return F.pad(img, padding).half()
    return F.pad(img, padding)


def unpad_image(img, padding):
    if padding == (0, 0, 0, 0):
        return img
    return img[: img.shape[0] - padding[3], : img.shape[1] - padding[1]]


def frame_to_tensor(frame, device):
    tensor = (
        torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    return tensor


def tensor_to_frame(tensor, w, h, fp16=False):
    if not fp16:
        frame = (tensor[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
    else:
        frame = (
            (tensor[0].float() * 255.0)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
    return frame[:h, :w]
