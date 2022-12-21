import _thread
import argparse
import os
import subprocess as sp
import sys
import time
import warnings
from queue import Empty, Queue
from threading import Event

import cv2
import numpy as np
import skvideo.io
import torch
from torch.nn import functional as F
from tqdm import tqdm
from win32api import GetShortPathName as ShortName

from logger import print
from model.pytorch_msssim import SSIM_Matlab, ssim_matlab

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Interpolation for video")
parser.add_argument(
    "video",
    metavar="VIDEO_PATH",
    type=str,
    default=None,
    help="Path to video file to be interpolated",
)
parser.add_argument(
    "--model",
    dest="modelDir",
    type=str,
    default="train_log",
    help="directory with trained model files",
)
parser.add_argument("--UHD", dest="UHD", action="store_true", help="support 4k video")
parser.add_argument(
    "--scale", dest="scale", type=float, default=1.0, help="Try scale=0.5 for 4k video"
)
parser.add_argument(
    "--ext", dest="ext", type=str, default="mp4", help="Output video extension"
)
parser.add_argument(
    "--multi", dest="multi", type=int, default=2, help="Target FPS multipiler"
)
parser.add_argument(
    "--ssim",
    dest="ssim",
    type=float,
    default=0.4,
    help="SSIM threshold for detect scene switching, larger num means more sensitive",
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
parser.add_argument(
    "--stop_time",
    dest="stop_time",
    type=float,
    default=-1,
    help="Stop time in seconds, will disable audio copy",
)
parser.add_argument(
    "--start_point",
    dest="start_point",
    type=int,
    default=0,
    help="Set process start point if you want to skip some frame, will disable audio copy",
)
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Show ffmpeg output for debugging",
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
print("Using device: {}".format(device))

try:
    from train_log.RIFE_HDv3 import Model
except:
    print("Please download our model from model list")
model = Model()
if not hasattr(model, "version"):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded model.")
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
target_fps = fps * args.multi
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoCapture.release()
print(
    f"Input info:{tot_frame} frames in total, {width}x{height}, {fps} fps to {target_fps} fps"
)
assert args.start_point < tot_frame, "Start point should be smaller than total frame"

tot_frame -= args.start_point
if args.stop_time > 0:
    tot_frame = min(int(args.stop_time * fps), tot_frame)
tot_frame = int(tot_frame)
print(f"{tot_frame} frames to process")

videogen = skvideo.io.vreader(args.video)
frame_count = 0
if args.start_point > 0:
    print("Skipping frames...")
    for _ in tqdm(range(args.start_point - 1)):
        next(videogen)
    print("Continue processing...")
lastframe = next(videogen)
video_path_wo_ext, ext = os.path.splitext(args.video)
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
proc = None

if args.start_point == 0:
    vid_out_name = f"{video_path_wo_ext}_{args.multi}X_{int(np.round(target_fps))}fps_noaudio.{args.ext}"
else:
    vid_out_name = f"{video_path_wo_ext}_{args.multi}X_{int(np.round(target_fps))}fps_noaudio_from{args.start_point}.{args.ext}"

if args.no_compression:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1 / mp4v / I420(raw)
    vid_out = cv2.VideoWriter(
        vid_out_name,
        fourcc,
        target_fps,
        (w, h),
        (cv2.VIDEO_ACCELERATION_ANY | cv2.VIDEOWRITER_PROP_HW_ACCELERATION),
    )
    assert vid_out.isOpened(), "Cannot open video for writing"
    print("Output video without compression")
else:
    if h * w > 9437184 and "h264" in args.encoder:
        print(
            "Warning: frame size reached h264 encoder upper limit (4096x2304), switching to HEVC encoder"
        )
        args.encoder = "libx265"
    if os.path.exists(vid_out_name):
        os.remove(vid_out_name)
    with open(vid_out_name, "w") as f:
        pass
    origin_file = ShortName(os.path.abspath(args.video))
    quality_option = "-crf" if "lib" in args.encoder else "-q:v"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{int(w)}x{int(h)}",
        "-pix_fmt", "bgr24", "-r", f"{target_fps}",
        "-i", "-",
        "-c:v", args.encoder, quality_option, str(args.crf),
        ShortName(vid_out_name),
    ]  # fmt: skip
    if args.debug:
        print(f"FFmpeg command: {' '.join(command)}")
    proc = sp.Popen(command, stdin=sp.PIPE, shell=False)
    print("FFmpeg backend initialized")

if (not args.no_compression) and args.stop_time <= 0 and args.start_point == 0:
    print("Audio will be copied from original video after processing is done")

running = True
stop_flag = False


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
    return F.pad(img, padding)


tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
I1 = (
    torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
    .to(device, non_blocking=True)
    .unsqueeze(0)
    .float()
    / 255.0
)
I1 = pad_image(I1)
temp = None  # save lastframe when processing static frame

empty_frame = np.zeros((180, 530, 3), dtype=np.uint8)
cv2.putText(
    empty_frame,
    "Preview disabled",
    (130, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    1,
)


show_empty = False
preview_origin = False
scenes = 0
ssim = 0
ssim_sum = 0
ssim_cnt = 0
last_scene = 0
last_frame = empty_frame.copy()
skipping_event = Event()
buffer_size_write = 64 if args.UHD else 256
buffer_size_read = 32
target_short_side = 520
target_long_side = 960
write_buffer = Queue(maxsize=buffer_size_write)
read_buffer = Queue(maxsize=buffer_size_read)


def show_frame(
    frame, frame_id, multi, total_frame, in_queue_write, in_queue_read
) -> None:
    global show_empty, preview_origin
    global empty_frame, last_frame
    global scenes, last_scene
    global target_short_side, target_long_side
    global stop_flag
    global skipping_event
    global ssim, ssim_sum, ssim_cnt

    if frame is None:
        return
    if (preview_origin and frame_id % multi == 0) or (
        not preview_origin and frame_id % multi == 1
    ):
        if not show_empty:
            # short_side = min(frame.shape[:2])
            # ratio = target_short_side / short_side
            long_side = max(frame.shape[:2])
            ratio = target_long_side / long_side
            temp_frame = cv2.resize(
                frame,
                (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            temp_frame = empty_frame.copy()
        height = temp_frame.shape[0]
        width = temp_frame.shape[1]
        progress = frame_id / total_frame
        text1 = (
            "Previewing original frame"
            if preview_origin
            else "Previewing interpolated frame"
        )
        ssim_avg = ssim_sum / ssim_cnt if ssim_cnt > 0 else -1
        text2 = f"Scene={scenes} ssim={ssim:.4f}({ssim_avg:.3f})"
        video_file_size = os.path.getsize(vid_out_name) / 1024 / 1024
        frame_time = frame_id / target_fps
        total_time = total_frame / target_fps
        time_str = f"{int(frame_time / 60):02d}:{frame_time % 60:04.1f}/{int(total_time / 60):02d}:{int(total_time % 60):02d}"
        text3 = (
            f"Write:{video_file_size:.2f}MB {time_str}"
            if video_file_size < 1024
            else f"Write:{video_file_size/1024:.2f}GB {time_str}"
        )
        warning = False
        if in_queue_write > 6:
            text3 += f" (Write +{in_queue_write:d}" + (
                "*)" if in_queue_write >= buffer_size_write else ")"
            )
            warning = True
        if (
            in_queue_read < buffer_size_read - 6
            and total_frame - frame_id > buffer_size_read
        ):
            text3 += f" (Read -{buffer_size_read - in_queue_read:d})"
            warning = True
        text4 = f"Frame={int(frame_id):d}/{int(total_frame):d} {progress:.2%}"
        if skipping_event.is_set():
            skipping_event.clear()
            cv2.putText(
                temp_frame,
                "[Skipping congruous frames...]",
                (5, height - 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if scenes != last_scene:
            last_scene = scenes
            cv2.putText(
                temp_frame,
                "[New scene detected]",
                (5, height - 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (140, 220, 0),
                2,
            )
        cv2.putText(
            temp_frame,
            text1,
            (5, height - 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            temp_frame,
            text2,
            (5, height - 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            temp_frame,
            text3,
            (5, height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if warning else (0, 0, 255),
            2,
        )
        cv2.putText(
            temp_frame,
            text4,
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
        read_progress = (frame_id + in_queue_write + in_queue_read) / total_frame
        cv2.rectangle(
            temp_frame,
            (0, height - 10),
            (int(width * read_progress), height),
            (221, 202, 98),
            thickness=cv2.FILLED,
        )
        write_progress = (frame_id + in_queue_write) / total_frame
        cv2.rectangle(
            temp_frame,
            (0, height - 10),
            (int(width * write_progress), height),
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
    elif key == ord("w"):
        target_long_side *= 1.1
        target_short_side *= 1.1
    elif key == ord("s"):
        target_long_side *= 0.9
        target_short_side *= 0.9
        target_long_side = max(40, target_long_side)
        target_short_side = max(40, target_short_side)
    elif key == ord("p"):
        stop_flag = True


def write_worker(user_args, write_buffer, read_buffer, total_frame):
    frame_id = 0
    no_compression = user_args.no_compression
    multi = user_args.multi
    global running
    cv2.namedWindow("Frame preview", cv2.WINDOW_AUTOSIZE)
    while running:
        item = write_buffer.get()
        if item is None:
            break
        frame = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        try:
            if no_compression:
                vid_out.write(frame)
            else:
                proc.stdin.write(frame.tostring())
        except:
            print("Error writing frame to video")
            running = False
            return
        in_queue_write = write_buffer.qsize()
        in_queue_read = read_buffer.qsize()
        show_frame(frame, frame_id, multi, total_frame, in_queue_write, in_queue_read)
        frame_id += 1


def read_worker(read_buffer, videogen):
    try:
        for frame in videogen:
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


_thread.start_new_thread(read_worker, (read_buffer, videogen))
_thread.start_new_thread(
    write_worker, (args, write_buffer, read_buffer, tot_frame * args.multi)
)

pbar = tqdm(total=tot_frame)
end_point = None
try:
    ssim_c1 = SSIM_Matlab()
    ssim_c2 = SSIM_Matlab()
    SKIP_THRESHOLD = 0.9999
    while running:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None or frame_count >= tot_frame:
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
        # ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        ssim = ssim_c1.calc(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996 and ssim < SKIP_THRESHOLD:
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
            # ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            ssim = ssim_c2.calc(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < args.ssim:
            output = []
            scenes += 1
            # if args.multi == 2:
            #     for i in range(args.multi - 1):
            #         output.append(I0)
            # else:
            if True:
                step = 1 / args.multi
                alpha = 0
                for i in range(args.multi - 1):
                    alpha += step
                    beta = 1 - alpha
                    output.append(
                        torch.from_numpy(
                            np.transpose(
                                (
                                    cv2.addWeighted(
                                        frame[:, :, ::-1],
                                        alpha,
                                        lastframe[:, :, ::-1],
                                        beta,
                                        0,
                                    )[:, :, ::-1].copy()
                                ),
                                (2, 0, 1),
                            )
                        )
                        .to(device, non_blocking=True)
                        .unsqueeze(0)
                        .float()
                        / 255.0
                    )
        elif ssim > 0.9999:
            output = [I0 for _ in range(args.multi - 1)]
            skipping_event.set()
        else:
            ssim_sum += ssim
            ssim_cnt += 1
            output = make_inference(I0, I1, args.multi - 1)
        write_buffer.put(lastframe)
        for mid in output:
            mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid[:h, :w])
        frame_count += 1
        pbar.update(1)
        lastframe = frame
        if stop_flag:
            pbar.close()
            write_buffer.put(lastframe)
            end_point = frame_count + args.start_point
            print(f"Manually stopped, ending point: {end_point}")
            break
        if break_flag:
            break
    write_buffer.put(lastframe)  # write the last frame
    frame_count += 1
except KeyboardInterrupt:
    print(f"Force stop")
    running = False
    write_buffer.put(None)
    sys.exit(1)
pbar.close()
write_buffer.put(None)
t0 = time.time()
try:
    if running:
        print("Waiting for write buffer to be empty")
        while not write_buffer.empty():
            if time.time() - t0 > 200:
                raise TimeoutError("Write buffer wait timeout")
            time.sleep(1)
except Exception as e:
    print(f"Force release video writer ({e})")
    running = False
cv2.destroyAllWindows()

if args.no_compression:
    vid_out.release()
else:
    try:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
    except:
        pass
    proc.wait()

if end_point is None and args.stop_time > 0:
    end_point = frame_count + args.start_point
if end_point is not None:
    target_name = (
        os.path.splitext(vid_out_name)[0]
        + f"_to{end_point}"
        + os.path.splitext(vid_out_name)[1]
    )
    if os.path.exists(target_name):
        os.remove(target_name)
    os.rename(vid_out_name, target_name)
    vid_out_name = target_name

if (not args.no_compression) and end_point is None and args.start_point == 0:
    # merge audio from original video
    print("Merging audio")
    target_name = vid_out_name.replace("_noaudio", "")
    if os.path.exists(target_name):
        os.remove(target_name)
    open(target_name, "w").close()
    command = [
        "ffmpeg", "-y","-hide_banner", "-loglevel", "error",
        "-i", vid_out_name, "-i", origin_file,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        ShortName(target_name),
    ]  # fmt: skip
    if args.debug:
        print(f"FFmpeg command: {' '.join(command)}")
    failed = False
    try:
        sp.run(command, check=True)
    except:
        print("Failed to merge audio")
        print(f"FFmpeg command: {' '.join(command)}")
        failed = True
    if not failed:
        os.remove(vid_out_name)

print("Process finished")
