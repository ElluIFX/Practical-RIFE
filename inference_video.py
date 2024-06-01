import argparse
import os
import sys
import time
import warnings

import cv2
import torch
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm

import richuru
from model.pytorch_msssim import SSIM_Matlab
from model_utils import (
    calc_padding,
    frame_to_tensor,
    make_inference,
    montage,
    pad_image,
    tensor_to_frame,
)
from trained import DEFAULT_MODEL, MODEL_LIST, load_model
from utils import (
    FramePreviewWindow,
    ThreadedVideoReader,
    ThreadedVideoWriter,
    check_ffmepg_available_codec,
    check_ffmpeg_installed,
    ffmpeg_merge_video_and_audio,
    ffmpeg_merge_videos,
    find_unfinished_last_file,
    find_unfinished_merge_list,
    get_video_info,
)

warnings.filterwarnings("ignore")

richuru.install()
check_ffmpeg_installed()
codecs = check_ffmepg_available_codec()
enc = []
for codec in codecs.values():
    enc.extend(codec[0])
enc_def = (
    "h264_qsv"
    if "h264_qsv" in enc
    else ("h264_nvenc" if "h264_nvenc" in enc else "libx264")
)

parser = argparse.ArgumentParser(
    description="Interpolation for video",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "video",
    metavar="VIDEO_PATH",
    type=str,
    default=None,
    help="Path to video file to be interpolated",
)
parser.add_argument(
    "--output",
    dest="output",
    type=str,
    default=None,
    help="Output video file path (auto generate if not specified)",
)
parser.add_argument(
    "--multi", dest="multi", type=int, default=2, help="Target FPS multipiler"
)
parser.add_argument(
    "--uhd",
    dest="uhd",
    action="store_true",
    help="Processing high-res video (>=4K)",
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
    "--ssim",
    dest="ssim",
    type=float,
    default=0.4,
    help="SSIM threshold for detect scene switching, larger num means more sensitive",
)
parser.add_argument(
    "--scene_copy",
    dest="scene_copy",
    action="store_true",
    help="Copy last frame when scene switching detected, instead of stacking frames",
)
parser.add_argument(
    "--codec",
    dest="codec",
    type=str,
    default=enc_def,
    choices=enc,
    help="Codec for ffmpeg backend encoding",
)
parser.add_argument(
    "--ext", dest="ext", type=str, default="mp4", help="Output video extension"
)
parser.add_argument(
    "--quality",
    dest="quality",
    type=int,
    default=17,
    help="Compression factor for h264/hevc codec, smaller is better quality",
)
parser.add_argument(
    "--start_frame",
    dest="start_frame",
    type=int,
    default=0,
    help="Process from specific frame, will disable audio copy if > 1",
)
parser.add_argument(
    "--start_time",
    dest="start_time",
    type=float,
    default=0,
    help="Start time in seconds, will disable audio copy",
)
parser.add_argument(
    "--stop_time",
    dest="stop_time",
    type=float,
    default=-1,
    help="Stop time in seconds, will disable audio copy",
)
parser.add_argument(
    "--skip_frame",
    dest="skip_frame",
    type=int,
    default=0,
    help="Skip N frames per frame when reading original video (original_fps /= (1 + N))",
)
parser.add_argument(
    "--resize",
    dest="resize",
    type=float,
    default=1.0,
    help="Resize input video to specific scale factor (0<resize<1)",
)
parser.add_argument(
    "--headless",
    dest="headless",
    action="store_true",
    help="Disable preview window",
)
parser.add_argument(
    "--raw_cmp",
    dest="raw_cmp",
    action="store_true",
    help="Split processed video for comparison (L: processed, R: raw)",
)
parser.add_argument(
    "--model_cmp",
    dest="model_cmp",
    default=None,
    choices=MODEL_LIST + [None],
    help="Another model for split comparison, 2x time cost (L: model, R: model_cmp)",
)
parser.add_argument(
    "--cmp_mode",
    dest="cmp_mode",
    default="L-R",
    choices=["L-R", "center", "left", "right"],
    help="Comparison view mode",
)
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Write debug log to stdout",
)
parser.add_argument(
    "--force_fp16",
    dest="force_fp16",
    action="store_true",
    help="Force fp16 mode even if input is too large",
)
args = parser.parse_args()
# logger.remove()
# logger.add(sys.stderr, level="INFO" if not args.debug else "DEBUG")
richuru.install(level="INFO" if not args.debug else "DEBUG", rich_traceback=False)
logger.add("inference.log", level="DEBUG", encoding="utf-8", rotation="1 MB")

if args.uhd and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0], "Invalid scale value"

logger.info(f'Input video: "{args.video}"')
fps, tot_frame, width, height = get_video_info(args.video)
args.model = args.model.upper()

raw_fps = fps
raw_tot_frame = tot_frame
raw_width, raw_height = width, height

if args.resize < 1.0:
    width = int(width * args.resize)
    height = int(height * args.resize)
    logger.info(f"Resize enabled, {raw_width}x{raw_height} => {width}x{height}")

if args.skip_frame > 0:
    tot_frame = int(tot_frame / (1 + args.skip_frame))
    fps = fps / (1 + args.skip_frame)
    logger.info(
        f"Skip frame enabled, FPS: {raw_fps} => {fps}, Total frame: {raw_tot_frame} => {tot_frame}"
    )

target_fps = fps * args.multi
logger.info(
    f"Input format: {tot_frame} frames, {width}x{height}, {fps} fps => {target_fps} fps"
)
if args.scale >= 1 and width * height > 1920 * 1080 * 2:
    logger.warning(
        "Input video is a high-res video, consider add --uhd option if processing is slow or failed"
    )
if args.start_time > 0:
    args.start_frame = round(args.start_time * raw_fps)
    assert (
        args.start_frame < tot_frame
    ), f"Start time should be smaller than total time ({tot_frame / raw_fps} sec)"
assert (
    args.start_frame < tot_frame
), f"Start frame should be smaller than total frame ({tot_frame})"


if args.output is not None:
    try:
        args.ext = os.path.splitext(args.output)[1][1:]
    except IndexError:
        pass
    video_path_prefix = os.path.splitext(args.output)[0] + "_noaudio"
else:
    video_path_wo_ext = os.path.splitext(args.video)[0]
    video_path_prefix = (
        f"{video_path_wo_ext}_{args.model}_{args.multi}X_{round(target_fps)}fps_noaudio"
    )
video_path = f"{video_path_prefix}.{args.ext}"


continue_process = False
if args.start_frame == 0:
    last_file, last_num = find_unfinished_last_file(video_path)
    if last_file is not None:
        logger.success("Found unfinished file, continue processing...")
        continue_process = True
        args.start_frame = last_num + 1
if args.start_frame > 0:
    video_path = f"{video_path_prefix}_from{args.start_frame}.{args.ext}"


tot_frame -= args.start_frame
if args.stop_time > 0:
    tot_frame = min(int(args.stop_time * raw_fps), tot_frame)
tot_frame = int(tot_frame)
logger.info(f"{tot_frame} frames to process")

if (height > 2000 or width > 2000) and args.fp16 and not args.force_fp16:
    logger.info("FP16 is disabled for input larger than 2000x2000")
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

if args.model_cmp is not None:
    assert args.model_cmp != args.model, "Comparison model should be different"
    model_cmp = load_model(args.model_cmp)
    args.raw_cmp = False
    logger.success(f"Loaded comparison model: {args.model_cmp}")
else:
    model_cmp = None
cmp_lt = f"{args.model} model" if args.model_cmp else "Processed"
cmp_rt = f"{args.model_cmp} model" if args.model_cmp else "Original"

scenes = []
skips = []
ssim = 0
ssim_sum = 0.0
ssim_cnt = 0


def notify_callback(frame_id: int):
    global ssim, ssim_sum, ssim_cnt
    ssim_avg = ssim_sum / ssim_cnt if ssim_cnt > 0 else -1
    info = [f"Scene={len(scenes)} ssim={ssim:.4f}({ssim_avg:.3f})"]
    notify = None
    if frame_id // args.multi in skips:
        notify = "[Skipping congruous frames...]"
    if frame_id // args.multi in scenes:
        notify = "[New scene detected]"
    return info, notify


writer = ThreadedVideoWriter(
    video_path,
    width,
    height,
    target_fps,
    codec=args.codec,
    quality=args.quality,
    convert_rgb=True,
    buffer_size=64 if args.uhd else 256,
)
reader = ThreadedVideoReader(
    args.video,
    start_time=args.start_frame / raw_fps,
    skip_frame=args.skip_frame,
    resize=(width, height) if args.resize < 1.0 else None,
)  # inputdict={"-hwaccel": "d3d11va"} dxva2 / cuda / cuvid / d3d11va / qsv / opencl

if not args.headless:
    preview = FramePreviewWindow(
        reader=reader,
        writer=writer,
        total_frame=tot_frame * args.multi,
        fps=target_fps,
        file_name=video_path,
    )

    preview.reg_notify_callback(notify_callback)
    preview.add_source("Original frame", lambda id: id % args.multi == 0)
    preview.add_source(
        "Interpolated frame", lambda id: id % args.multi == 1, default=True
    )

if args.stop_time <= 0 and args.start_frame == 0:
    logger.info("Audio will be copied from original video after processing is done")

end_point = None
frame_count = 0
lastframe = reader.get()
if lastframe is None:
    logger.error("Failed to read first frame from video source")
    exit(1)
assert (
    lastframe.shape[1] == width and lastframe.shape[0] == height
), f"Invalid frame size: {lastframe.shape[1]}x{lastframe.shape[0]}"

padding = calc_padding(width, height, args.scale)
I1 = frame_to_tensor(lastframe, device)
I1 = pad_image(I1, padding, args.fp16)

time.sleep(1)
pbar = tqdm(total=tot_frame)
try:
    ssim_c = SSIM_Matlab()
    SKIP_THRESHOLD = 0.9999
    while True:
        frame = reader.get()
        if frame is None or frame_count >= tot_frame:
            break
        assert (
            frame.shape[1] == width and frame.shape[0] == height
        ), f"Invalid frame size: {frame.shape[1]}x{frame.shape[0]}"
        I0 = I1
        I1 = frame_to_tensor(frame, device)
        I1 = pad_image(I1, padding, args.fp16)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = float(ssim_c.calc(I0_small[:, :3], I1_small[:, :3]))  # type: ignore

        if ssim < args.ssim:
            output = []
            scenes.append(frame_count)
            if args.scene_copy:  # copy last frame
                for i in range(args.multi - 1):
                    output.append(I0)
            else:  # stack frames
                step = 1 / args.multi
                alpha = 0
                for _ in range(args.multi - 1):
                    alpha += step
                    beta = 1 - alpha
                    mid_frame = cv2.addWeighted(  # type: ignore
                        frame[:, :, ::-1],
                        alpha,
                        lastframe[:, :, ::-1],
                        beta,
                        0,
                    )[:, :, ::-1].copy()
                    output.append(mid_frame)
            cmp_output = output
        elif ssim >= SKIP_THRESHOLD:
            output = [lastframe for _ in range(args.multi - 1)]
            cmp_output = output
            skips.append(frame_count)
        else:
            ssim_sum += ssim
            ssim_cnt += 1
            infs = make_inference(model, I0, I1, args.multi - 1, args.scale)
            output = [tensor_to_frame(inf, width, height, args.fp16) for inf in infs]
            if model_cmp:
                cmp_infs = make_inference(model_cmp, I0, I1, args.multi - 1, args.scale)
                cmp_output = [
                    tensor_to_frame(inf, width, height, args.fp16) for inf in cmp_infs
                ]
        if args.raw_cmp or args.model_cmp:
            writer.put(montage(lastframe, lastframe, args.cmp_mode, cmp_lt, cmp_rt))
        else:
            writer.put(lastframe)
        for mid in output:
            if args.raw_cmp:
                writer.put(montage(mid, lastframe, args.cmp_mode, cmp_lt, cmp_rt))
            elif args.model_cmp:
                writer.put(
                    montage(mid, cmp_output.pop(0), args.cmp_mode, cmp_lt, cmp_rt)
                )
            else:
                writer.put(mid)
        frame_count += 1
        pbar.update(1)
        lastframe = frame
        if not args.headless and preview.is_stopped:
            pbar.close()
            writer.put(lastframe)
            end_point = frame_count + args.start_frame
            logger.warning(f"Manually stopped, ending point: {end_point}")
            break
    writer.put(lastframe)  # write the last frame
    frame_count += 1
except KeyboardInterrupt:
    pbar.close()
    end_point = frame_count + args.start_frame
    logger.critical(f"[UNSAFE] Manually stopped, ending point: {end_point}")
pbar.close()
writer.close()
reader.close()
if not args.headless:
    preview.close()

logger.debug(f"Processed: {frame_count} frames, {len(scenes)} scenes detected")

if end_point is None and args.stop_time > 0:
    end_point = frame_count + args.start_frame
if end_point is not None:
    target_name = (
        os.path.splitext(video_path)[0]
        + f"_to{end_point}"
        + os.path.splitext(video_path)[1]
    )
    if os.path.exists(target_name):
        os.remove(target_name)
    os.rename(video_path, target_name)
else:
    if continue_process:
        logger.info("Merging videos")
        video_path = f"{video_path_prefix}.{args.ext}"
        merge_list = find_unfinished_merge_list(video_path)
        if merge_list is not None:
            if ffmpeg_merge_videos(video_path, merge_list):
                args.start_frame = 0  # trick to trigger the next if

    if args.start_frame <= 1:
        logger.info("Merging audio")
        target_name = video_path.replace("_noaudio", "")
        if ffmpeg_merge_video_and_audio(
            target_name, video_path, os.path.abspath(args.video)
        ):
            os.remove(video_path)
logger.success("Process finished")
sys.exit(0)
