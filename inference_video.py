import _thread
import argparse
import os
import subprocess as sp
import time
import warnings
from queue import Empty, Queue

import cv2
import numpy as np
import skvideo.io
import torch
from torch.nn import functional as F
from tqdm import tqdm
from win32api import GetShortPathName as ShortName

from logger import print
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Interpolation for video")
parser.add_argument("--video", dest="video", type=str, default=None)
parser.add_argument(
    "--output",
    dest="output",
    type=str,
    default=None,
    help="If not specified, output name will be automatically generated",
    required=True,
)
parser.add_argument(
    "--montage", dest="montage", action="store_true", help="montage origin video"
)
parser.add_argument(
    "--model",
    dest="modelDir",
    type=str,
    default="train_log",
    help="directory with trained model files",
)
parser.add_argument(
    "--fp16",
    dest="fp16",
    action="store_true",
    help="fp16 mode for faster and more lightweight inference on cards with Tensor Cores",
)
parser.add_argument("--UHD", dest="UHD", action="store_true", help="support 4k video")
parser.add_argument(
    "--scale", dest="scale", type=float, default=1.0, help="Try scale=0.5 for 4k video"
)
parser.add_argument(
    "--fps",
    dest="fps",
    type=int,
    default=None,
    help="Target fps of output video, use --multi instead",
)
parser.add_argument(
    "--ext", dest="ext", type=str, default="mp4", help="Output video extension"
)
parser.add_argument(
    "--multi", dest="multi", type=int, default=2, help="Target FPS multiple"
)
parser.add_argument(
    "--no_compression",
    dest="no_compression",
    action="store_true",
    help="Disable ffmpeg backend for video compression, causes output no audio",
)
parser.add_argument(
    "--encoder",
    dest="encoder",
    type=str,
    default="h264_qsv",
    help="Encoder for ffmpeg",
)
parser.add_argument(
    "--crf",
    dest="crf",
    type=int,
    default=17,
    help="Compression factor for h264 encoder",
)
args = parser.parse_args()
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
print("Using device: {}".format(device))

try:
    from train_log.RIFE_HDv3 import Model
except:
    print("Please download our model from model list")
model = Model()
if not hasattr(model, "version"):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()
if args.fps is None:
    fpsNotAssigned = True
    args.fps = fps * args.multi
else:
    fpsNotAssigned = False
videogen = skvideo.io.vreader(args.video)
lastframe = next(videogen)
fourcc = cv2.VideoWriter_fourcc(*"avc1")  # avc1 / mp4v
video_path_wo_ext, ext = os.path.splitext(args.video)
print(
    "{}.{}, {} frames in total, {}FPS to {}FPS".format(
        video_path_wo_ext,
        args.ext,
        tot_frame,
        fps,
        args.fps,
    )
)
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
proc = None
if args.output is not None:
    vid_out_name = args.output
else:
    vid_out_name = "{}_{}X_{}fps.{}".format(
        video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext
    )
    if args.no_compression:
        vid_out = cv2.VideoWriter(
            vid_out_name,
            fourcc,
            args.fps,
            (w, h),
            (cv2.VIDEO_ACCELERATION_ANY | cv2.VIDEOWRITER_PROP_HW_ACCELERATION),
        )
        assert vid_out.isOpened(), "Cannot open video for writing"
        print("Output video without compression")
    else:
        if os.path.exists(vid_out_name):
            os.remove(vid_out_name)
        with open(vid_out_name, "w") as f:
            pass
        origin_file = ShortName(os.path.abspath(args.video))
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{args.fps}",
            "-i",
            "-",
            "-i",
            origin_file,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            args.encoder,
            "-q:v",
            str(args.crf),
            "-c:a",
            "copy",
            ShortName(vid_out_name),
        ]
        # proc = sp.Popen(command, stdin=sp.PIPE, shell=True)
        proc = sp.Popen(
            command, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.DEVNULL, shell=True
        )
        print("FFmpeg backend initialized")
print("Video resolution: {}x{}".format(int(w), int(h)))


def clear_write_buffer(user_args, write_buffer, total_frame):
    cnt = 0
    cv2.namedWindow("Frame preview", cv2.WINDOW_AUTOSIZE)
    while True:
        item = write_buffer.get()
        if item is None:
            break
        frame = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        cnt += 1
        if user_args.no_compression:
            vid_out.write(frame)
        else:
            proc.stdin.write(frame.tostring())
        in_queue = write_buffer.qsize()
        show_frame(frame, cnt, total_frame, in_queue)


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if user_args.montage:
                frame = frame[:, left : left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1, n):
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n // 2)
        second_half = make_inference(middle, I1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def pad_image(img):
    if args.fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left : left + w]
buffer_size = 64 if args.UHD else 256
write_buffer = Queue(maxsize=buffer_size)
read_buffer = Queue(maxsize=32)

I1 = (
    torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
    .to(device, non_blocking=True)
    .unsqueeze(0)
    .float()
    / 255.0
)
I1 = pad_image(I1)
temp = None  # save lastframe when processing static frame
target_height = 580

empty_frame = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.putText(
    empty_frame, "No Preview", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1
)


show_empty = False
preview_origin = False
last_frame = empty_frame.copy()


def show_frame(frame, frame_id, total_frame, in_queue) -> None:
    global show_empty
    global empty_frame
    global preview_origin
    global last_frame
    target_short_side = 520
    target_long_side = 960
    if frame is None:
        return
    if show_empty:
        cv2.imshow("Frame preview", empty_frame)
    else:
        if (preview_origin and frame_id % 2 == 1) or (
            not preview_origin and frame_id % 2 == 0
        ):
            # short_side = min(frame.shape[:2])
            # ratio = target_short_side / short_side
            long_side = max(frame.shape[:2])
            ratio = target_long_side / long_side
            last_frame = cv2.resize(
                frame,
                (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
                interpolation=cv2.INTER_NEAREST,
            )
        temp_frame = last_frame.copy()
        height = temp_frame.shape[0]
        width = temp_frame.shape[1]
        progress = frame_id / total_frame
        text = (
            "Previewing original frame"
            if preview_origin
            else "Previewing interpolated frame"
        )
        cv2.putText(
            temp_frame,
            text,
            (5, height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        text = f"Frame={int(frame_id):d}/{int(total_frame):d} {progress:.2%}"
        if in_queue > 6:
            text += (
                f" (Overflow +{in_queue:d}"
                + ("*" if in_queue >= buffer_size else "")
                + ")"
            )
        cv2.putText(
            temp_frame,
            text,
            (5, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.rectangle(
            temp_frame,
            (0, height - 10),
            (width, height),
            (0, 0, 0),
            thickness=cv2.FILLED,
        )
        if in_queue > 6:
            over_progress = (frame_id + in_queue) / total_frame
            cv2.rectangle(
                temp_frame,
                (0, height - 10),
                (int(width * over_progress), height),
                (0, 255, 255),
                thickness=cv2.FILLED,
            )
        cv2.rectangle(
            temp_frame,
            (0, height - 10),
            (int(width * progress), height),
            (0, 0, 255),
            thickness=cv2.FILLED,
        )
        cv2.imshow("Frame preview", temp_frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        show_empty = not show_empty
    elif key == 32:  # SPACE
        preview_origin = not preview_origin


_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer, tot_frame * 2))

try:
    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get()  # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = (
                torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                .to(device, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, args.scale)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.3:
            output = []
            for i in range(args.multi - 1):
                output.append(I0)
            """
            output = []
            step = 1 / args.multi
            alpha = 0
            for i in range(args.multi - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            """
        else:
            output = make_inference(I0, I1, args.multi - 1)

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break
    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
except KeyboardInterrupt:
    pass


pbar.close()
write_buffer.put(None)
print("waiting for write buffer to be empty")
while not write_buffer.empty():
    time.sleep(0.1)
cv2.destroyAllWindows()

if args.no_compression:
    vid_out.release()
else:
    try:
        proc.stdin.close()
        proc.stderr.close()
    except:
        pass
    proc.wait()
print("Process finished")
