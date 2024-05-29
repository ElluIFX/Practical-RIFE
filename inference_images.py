import argparse
import os
import shutil
import warnings

import cv2
import richuru
import torch
from loguru import logger
from tqdm import tqdm

from model_utils import (
    calc_padding,
    frame_to_tensor,
    make_inference,
    pad_image,
    tensor_to_frame,
)
from trained import DEFAULT_MODEL, MODEL_LIST, load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Interpolation for images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "input",
    metavar="IMAGE_PATH",
    type=str,
    default=None,
    help="Path to directory containing image files",
)
parser.add_argument(
    "--output",
    dest="output",
    type=str,
    default=None,
    help="Output directory path (auto generate if not specified)",
)
parser.add_argument(
    "--multi", dest="multi", type=int, default=2, help="Target FPS multipiler"
)
parser.add_argument(
    "--scale",
    dest="scale",
    type=float,
    default=1.0,
    choices=[0.25, 0.5, 1.0, 2.0, 4.0],
    help="Inferece scale factor, smaller is less usage",
)
parser.add_argument(
    "--model",
    dest="model",
    type=str,
    default=DEFAULT_MODEL,
    choices=MODEL_LIST,
    help="Model version to use for interpolation (L: Lite)",
)
parser.add_argument(
    "--fp16",
    dest="fp16",
    action="store_true",
    help="fp16 mode for faster and more lightweight inference on cards with Tensor Cores (if available)",
)
parser.add_argument(
    "--ext",
    dest="ext",
    type=str,
    default="png",
    help="Output image format",
    choices=["png", "jpg"],
)
parser.add_argument(
    "--quality",
    dest="quality",
    type=int,
    default=95,
    help="Quality of output image (.jpg only)",
)
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Write debug log to stdout",
)
parser.add_argument(
    "--loop",
    dest="loop",
    action="store_true",
    help="Loop the output sequence (Add first frame to the end)",
)
args = parser.parse_args()
richuru.install(level="INFO" if not args.debug else "DEBUG")

assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0], "Invalid scale value"

logger.info(f'Input image path: "{args.input}"')


def to_int(x: str) -> int:
    # find the rightmost digit in the string
    state = 0
    j = None
    for i in range(len(x) - 1, -1, -1):
        if not x[i].isdigit():
            if state == 1:
                if j is not None:
                    return int(x[i + 1 : j + 1])
                else:
                    return int(x[i + 1 :])
        else:
            if state == 0:
                state = 1
                j = i
    return -1


image_list = [
    os.path.join(args.input, f)
    for f in sorted(os.listdir(args.input), key=to_int)
    if f.endswith((".png", ".jpg", ".jpeg"))
]
assert len(image_list) > 0, "No image files found in input directory"
logger.info(f"Found {len(image_list)} images")
if not args.output:
    args.output = os.path.join(args.input, "inference_output")
if not os.path.exists(args.output):
    os.makedirs(args.output)
elif os.path.isdir(args.output):
    shutil.rmtree(args.output)
    os.makedirs(args.output)
assert os.path.isdir(args.output), "Output path is not a directory"

lastframe = cv2.imread(image_list[0])
width, height = lastframe.shape[1], lastframe.shape[0]
logger.info(f"Image resolution: {width}x{height}")
if (height > 2000 or width > 2000) and args.fp16:
    logger.warning("FP16 is disabled for input larger than 2000x2000")
    args.fp16 = False

if not torch.cuda.is_available():
    logger.error("CUDA is not available for PyTorch")
    exit(1)
device = torch.device("cuda")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)  # type: ignore
model = load_model(args.model)
logger.success(f"Loaded {args.model} model (version {model.version})")

padding = calc_padding(width, height, args.scale)
I1 = frame_to_tensor(lastframe, device)
I1 = pad_image(I1, padding, args.fp16)

if args.loop:
    image_list.append(image_list[0])


def save_image(image, idx):
    path = os.path.join(args.output, f"{idx:04d}.{args.ext}")
    if args.ext == "png":
        cv2.imwrite(path, image)
    elif args.ext == "jpg":
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])


idx = 0
for i, image_path in enumerate(tqdm(image_list[1:], desc="Processing")):
    I0 = I1
    frame = cv2.imread(image_path)
    assert (
        frame.shape[0] == height and frame.shape[1] == width
    ), f"Image resolution mismatch (expected {width}x{height} but got {frame.shape[1]}x{frame.shape[0]})"
    I1 = frame_to_tensor(frame, device)
    I1 = pad_image(I1, padding, args.fp16)
    infs = make_inference(model, I0, I1, args.multi - 1, args.scale)
    output = [tensor_to_frame(inf, width, height, args.fp16) for inf in infs]
    save_image(lastframe, idx)
    idx += 1
    for img in output:
        save_image(img, idx)
        idx += 1
    lastframe = frame
if not args.loop:
    save_image(lastframe, idx)
    idx += 1

logger.success(f'Saved {idx} images to "{args.output}", process completed')
