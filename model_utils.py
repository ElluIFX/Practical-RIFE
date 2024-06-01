from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from trained import MODEL


def make_inference(model: "MODEL", I0, I1, n, scale):
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


def frame_to_tensor(frame: np.ndarray, device) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    return tensor


def tensor_to_frame(tensor: torch.Tensor, w, h, fp16=False):
    if not fp16:
        frame = tensor[0] * 255.0
    else:
        frame = (tensor[0].float() * 255.0).clamp(0, 255)
    return frame.byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]


def montage(
    frame_l: np.ndarray,
    frame_r: np.ndarray,
    mode: Literal["L-R", "center", "left", "right"],
    left_text: str = "",
    right_text: str = "",
    draw_split_line: bool = True,
) -> np.ndarray:
    w = frame_l.shape[1]
    assert w == frame_r.shape[1]

    if mode == "L-R":
        new = np.concatenate((frame_l[:, : w // 2], frame_r[:, w // 2 :]), axis=1)
    elif mode == "center":
        l_w = w // 2
        r_w = w - l_w
        pad = r_w // 2
        new = np.concatenate(
            (frame_l[:, pad : l_w + pad], frame_r[:, pad : pad + r_w]),
            axis=1,
        )
    elif mode == "left":
        l_w = w // 2
        r_w = w - l_w
        new = np.concatenate((frame_l[:, :l_w], frame_r[:, :r_w]), axis=1)
    elif mode == "right":
        l_w = w // 2
        r_w = w - l_w
        new = np.concatenate((frame_l[:, -l_w:], frame_r[:, -r_w:]), axis=1)
    new = cv2.cvtColor(new, cv2.COLOR_RGB2BGR)
    if draw_split_line:
        if new.shape[2] == 3:
            new[:, w // 2] = (0, 0, 255)
        else:
            cv2.line(new, (w // 2, 0), (w // 2, new.shape[0]), (0, 0, 255), 1)
    if left_text:
        cv2.putText(
            new,
            left_text,
            (
                w // 2
                - 10
                - cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0],
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    if right_text:
        cv2.putText(
            new,
            right_text,
            (w // 2 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    return cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
