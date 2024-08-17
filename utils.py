import _thread
import json
import os
import re
import subprocess as sp
import sys
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skvideo.io
from loguru import logger
from tqdm import tqdm


class FFmpegWriter:
    def __init__(
        self,
        filename: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        quality: int = 17,
        frame_fmt: str = "bgr24",
        extra_opts: Dict[str, str] = {},
    ):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps

        if height * width > 4096 * 2304 and "264" in codec:
            new_codec = (
                "libx265" if codec == "libx264" else codec.replace("h264", "hevc")
            )
            logger.warning(
                f"Frame size reached {codec} codec limit (4096x2304), trying switch to {new_codec}"
            )
            codec = new_codec
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "w") as _:
            pass
        if "libx" in codec:
            quality_options = ["-crf", str(quality)]
        elif "qsv" in codec:
            quality_options = ["-q:v", str(quality)]
        elif "nvenc" in codec:
            if quality < 10:
                quality = 0  # lossless
            elif quality > 24:
                quality = 1000_000  # 1000kbps minimum
            else:
                quality = int((24 - quality) * 2000_000 + 1000_000)
            quality_options = ["-b:v", str(quality)]
        elif "amf" in codec:
            quality_options = [
                "-rc",
                "cqp",
                "-qp_i",
                str(quality - 1),
                "-qp_p",
                str(quality),
                "-qp_b",
                str(quality + 1),
            ]
        preset = "slow" if quality < 20 else "medium"
        pix_fmt = "yuv420p" if "264" in codec else "yuv444p"
        if "qsv" in codec:
            pix_fmt = "nv12"
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", frame_fmt, "-r", str(fps),
            "-i", "-",
            "-c:v", codec, *quality_options, "-preset", preset,
            "-pix_fmt", pix_fmt,
        ]  # fmt: skip
        if extra_opts:
            for k, v in extra_opts.items():
                command.extend([k, v])
        command.append(filename)
        logger.debug(f"FFmpeg backend command: {command}")
        self.proc = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        logger.success("FFmpeg backend initialized")

    def write_frame(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tostring())  # type: ignore

    def write_raw(self, raw: bytes) -> None:
        self.proc.stdin.write(raw)  # type: ignore

    def read_stdout(self) -> str:
        return self.proc.stdout.read().decode("utf-8", "ignore")  # type: ignore

    def read_stderr(self) -> str:
        return self.proc.stderr.read().decode("utf-8", "ignore")  # type: ignore

    def close(self, timeout=None) -> None:
        try:
            self.proc.stdin.close()  # type: ignore
            self.proc.stdout.close()  # type: ignore
            self.proc.stderr.close()  # type: ignore
        except Exception:
            pass
        self.proc.wait(timeout)
        logger.success("FFmpeg writer closed")


class ThreadedVideoWriter(FFmpegWriter):
    def __init__(
        self,
        filename: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        quality: int = 17,
        frame_fmt: str = "bgr24",
        extra_opts: Dict[str, str] = {},
        buffer_size: int = 32,
        callback: Optional[Callable[[np.ndarray, int], None]] = None,
        convert_rgb: bool = False,
    ):
        super().__init__(
            filename, width, height, fps, codec, quality, frame_fmt, extra_opts
        )
        self.callback = callback
        self.convert_rgb = convert_rgb
        self.buffer = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.running = True
        _thread.start_new_thread(self._worker, ())

    def close(self, timeout: int = 200) -> None:
        self.buffer.put(None)
        time.sleep(1)
        try:
            if not self.buffer.empty():
                logger.info("Waiting for writer buffer to be empty...")
                t0 = time.time()
                while not self.buffer.empty():
                    if time.time() - t0 > timeout:
                        raise TimeoutError("Writer buffer wait timeout")
                    time.sleep(1)
        except Exception:
            logger.warning("Force close writer")
            self.running = False
            return super().close(timeout=5)
        self.running = False
        return super().close()

    def _worker(self):
        frame_id = 0
        logger.success("Video writer started")
        while self.running:
            frame = self.buffer.get()
            if frame is None:
                break
            try:
                if self.convert_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.write_frame(frame)
                try:
                    if self.callback:
                        self.callback(frame, frame_id)
                except Exception:
                    logger.exception("Error occurred while callback")
                    self.callback = None
            except Exception:
                logger.exception("Error occurred while writing video")
                logger.debug(
                    f"ffmpeg output: \nerror:{self.read_stderr()}\nstdout:{self.read_stdout()}"
                )
                self.running = False
                break
            frame_id += 1
        logger.success("Video writer closed")

    @property
    def in_queue(self):
        return self.buffer.qsize()

    def put(self, frame: np.ndarray) -> None:
        self.buffer.put(frame)


class ThreadedVideoReader:
    def __init__(
        self,
        filename: str,
        start_time: float = 0,
        start_frame: int = 0,
        skip_frame: int = 0,
        skip_before_read: bool = False,
        inputdict: Optional[Dict[str, str]] = None,
        buffer_size: int = 32,
        resize: Optional[Tuple[int, int]] = None,
        resize_interpolation=cv2.INTER_AREA,
    ) -> None:
        self.filename = filename
        if start_time > 0:
            _inputdict = {"-ss": str(start_time)}  # may not accurate
            if inputdict:
                _inputdict.update(inputdict)
            videogen = skvideo.io.vreader(filename, inputdict=_inputdict)
            logger.info(f"Read video from {start_time}s")
        else:
            videogen = skvideo.io.vreader(filename, inputdict=inputdict)
        if start_frame > 0:
            logger.info(f"Skipping {start_frame - 1} frames...")
            for _ in tqdm(range(start_frame - 1)):
                next(videogen)
            logger.success("Continue reading")
        self.videogen = videogen
        self.buffer = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.skip_frame = skip_frame
        self.skip_before_read = skip_before_read
        self.resize = resize
        self.resize_interpolation = resize_interpolation
        self.running = True
        _thread.start_new_thread(self._worker, ())

    def close(self) -> None:
        self.running = False
        try:
            self.videogen.close()
        except ValueError:
            pass
        while not self.buffer.empty():
            self.buffer.get()

    def _worker(self):
        logger.success("Video reader started")
        try:
            if self.skip_frame == 0:
                for frame in self.videogen:
                    if not self.running:
                        break
                    if self.resize is not None:
                        frame = cv2.resize(
                            frame, self.resize, interpolation=self.resize_interpolation
                        )
                    self.buffer.put(frame)
            else:
                while self.running:
                    if self.skip_before_read:
                        for _ in range(self.skip_frame):
                            next(self.videogen)
                    frame = next(self.videogen)
                    if self.resize is not None:
                        frame = cv2.resize(
                            frame, self.resize, interpolation=self.resize_interpolation
                        )
                    self.buffer.put(frame)
                    if not self.skip_before_read:
                        for _ in range(self.skip_frame):
                            next(self.videogen)
        except (StopIteration, AssertionError, RuntimeError):
            pass
        except Exception as e:
            logger.exception(f"Error occurred while reading video: {e}")
        self.buffer.put(None)
        self.running = False
        logger.success("Video reader closed")

    def get(self) -> np.ndarray:
        return self.buffer.get()

    @property
    def in_queue(self):
        return self.buffer.qsize()


class FPSCounter:
    def __init__(self, max_sample=30) -> None:
        self.t = time.perf_counter()
        self.max_sample = max_sample
        self.t_list: List[float] = []

    def update(self) -> None:
        now = time.perf_counter()
        self.t_list.append(now - self.t)
        self.t = now
        if len(self.t_list) > self.max_sample:
            self.t_list.pop(0)

    @property
    def fps(self) -> float:
        length = len(self.t_list)
        sum_t = sum(self.t_list)
        if length == 0:
            return 0.0
        else:
            return length / sum_t

    def reset(self) -> None:
        self.t_list.clear()
        self.t = time.perf_counter()


class FramePreviewWindow:
    def __init__(
        self,
        reader: ThreadedVideoReader,
        writer: ThreadedVideoWriter,
        total_frame: int,
        fps: float,
        file_name: str,
    ):
        self._reader = reader
        self._writer = writer
        self._total_frame = total_frame
        self._fps = fps
        self._file_name = file_name
        self._sources = {}
        self._current_source = ""
        self._notify_callback = None
        self._show_empty = False
        self._empty_frame = np.zeros((210, 600, 3), dtype=np.uint8)
        cv2.putText(
            self._empty_frame,
            "Preview disabled",
            (130, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )
        self._target_short_side = 520
        self._target_long_side = 960
        self.running = True
        self._title = "Frame preview (ESC: ON/OFF, W/S: Zoom, P: Stop)"
        self._window_inited = False
        self._fpsc = FPSCounter()

        self.add_source("All output frame", lambda _: True, default=True)
        writer.callback = self._show_frame

    def add_source(
        self, name: str, condition: Callable[[int], bool], default: bool = False
    ):
        self._sources[name] = condition
        if default:
            self._current_source = name
        if len(self._sources) > 1:
            self._title = (
                "Frame preview (ESC: ON/OFF, SPACE: Switch source, W/S: Zoom, P: Stop)"
            )

    def reg_notify_callback(
        self, callback: Callable[[int], Tuple[List[str], Optional[str]]]
    ):
        self._notify_callback = callback

    @property
    def is_stopped(self):
        return not self.running

    def close(self):
        self.running = False
        cv2.destroyAllWindows()

    def _next_source(self):
        if len(self._sources) == 1:
            return
        sources = list(self._sources.keys())
        idx = sources.index(self._current_source)
        idx = (idx + 1) % len(sources)
        self._current_source = sources[idx]
        self._fpsc.reset()

    def _show_frame(self, frame: np.ndarray, frame_id: int) -> None:
        if frame is None or not self.running:
            return
        if not self._window_inited:
            self._window_inited = True
            cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
        if self._sources[self._current_source](frame_id):
            if self._notify_callback:
                extra_info, extra_notify = self._notify_callback(frame_id)
            else:
                extra_info, extra_notify = None, None
            in_queue_write = self._writer.in_queue
            in_queue_read = self._reader.in_queue
            buffer_size_write = self._writer.buffer_size
            buffer_size_read = self._reader.buffer_size
            if not self._show_empty:
                long_side = max(frame.shape[:2])
                ratio = self._target_long_side / long_side
                temp = cv2.resize(
                    frame,
                    (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                temp = self._empty_frame.copy()
            height, width = temp.shape[:2]
            y = height - 20
            x = 5
            offset = 22
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.6, 2
            info_color = (0, 0, 255)
            warn_color = (0, 255, 255)
            notify_color = (140, 220, 0)
            ############### Draw progress bar ###############
            progress = frame_id / self._total_frame
            pos = (0, height - 10)
            th = cv2.FILLED
            cv2.rectangle(temp, pos, (width, height), (0, 0, 0), th)
            read_progress = (
                frame_id + in_queue_write + in_queue_read
            ) / self._total_frame
            cv2.rectangle(
                temp, pos, (int(width * read_progress), height), (221, 202, 98), th
            )
            write_progress = (frame_id + in_queue_write) / self._total_frame
            cv2.rectangle(
                temp, pos, (int(width * write_progress), height), (0, 255, 255), th
            )
            cv2.rectangle(temp, pos, (int(width * progress), height), (0, 0, 255), th)
            ############### Progress text ###############
            self._fpsc.update()
            text = f"{int(frame_id):d}/{int(self._total_frame):d} {progress:.2%} {self._fpsc.fps:.2f}fps"
            cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            ############### Write info text ###############
            y -= offset
            video_file_size = os.path.getsize(self._file_name) / 1024 / 1024
            frame_time = frame_id / self._fps
            total_time = self._total_frame / self._fps
            time_str = f"{int(frame_time / 60):02d}:{frame_time % 60:04.1f}/{int(total_time / 60):02d}:{int(total_time % 60):02d}"
            text = (
                f"{video_file_size:.2f}/~{video_file_size/(frame_id/self._total_frame):.0f}MB {time_str}"
                if video_file_size < 1024
                else f"Write:{video_file_size/1024:.2f}GB {time_str}"
            )
            color = info_color
            if in_queue_write > 6 or in_queue_write > buffer_size_write / 2:
                if in_queue_write >= buffer_size_write:
                    text += " [Write Quene Full]"
                else:
                    text += " [Write Slow]"
                color = warn_color
            if in_queue_read < buffer_size_read - 6 and self._reader.running:
                text += " [Read Slow]"
                color = warn_color
            cv2.putText(temp, text, (x, y), font, scale, color, thickness)
            ############### Write extra info ###############
            if extra_info:
                for text in extra_info:
                    y -= offset
                    cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            ############### Write extra notify ###############
            if extra_notify:
                y -= offset
                cv2.putText(
                    temp, extra_notify, (x, y), font, scale, notify_color, thickness
                )
            ############### Write source text ###############
            if len(self._sources) > 1:
                y -= offset
                text = f"Source: {self._current_source}"
                cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            cv2.imshow(self._title, temp)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            self._show_empty = not self._show_empty
        elif key == 32:  # SPACE
            self._next_source()
        elif key == ord("w"):
            self._target_long_side *= 1.1
            self._target_short_side *= 1.1
        elif key == ord("s"):
            self._target_long_side *= 0.9
            self._target_short_side *= 0.9
            self._target_long_side = max(40, self._target_long_side)
            self._target_short_side = max(40, self._target_short_side)
        elif key == ord("p"):
            self.close()


@dataclass
class VideoInfo:
    width: int = -1
    height: int = -1
    codec: str = ""
    pix_fmt: str = ""
    frame_rate: float = -1
    duration: float = -1
    bit_rate: float = -1
    frames: int = -1
    audio_available: bool = False
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[float] = None
    audio_sample_rate: Optional[float] = None
    audio_channels: Optional[int] = None


def probe_video_info(video_path: str) -> Tuple[Optional[VideoInfo], Optional[Dict]]:
    command = [
        "ffprobe",
        "-loglevel",
        "quiet",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        os.path.abspath(video_path),
    ]

    try:
        read = (
            sp.check_output(command, stderr=sp.STDOUT)
            .decode("utf-8", "ignore")
            .strip()
            .replace("\n", "")
            .replace("\r", "")
        )
        raw: Dict = json.loads(read)
    except Exception as e:
        logger.error(f"Failed to read video info: {e}")
        return None, None
    if raw.get("streams") is None:
        return None, None
    info = VideoInfo()
    info.width = raw["streams"][0].get("width", -1)
    info.height = raw["streams"][0].get("height", -1)
    info.codec = raw["streams"][0].get("codec_name", "")
    info.pix_fmt = raw["streams"][0].get("pix_fmt", "")
    info.frame_rate = eval(raw["streams"][0].get("r_frame_rate", "-1"))
    if "format" in raw:
        info.duration = float(raw["format"].get("duration", -1))
        info.bit_rate = float(raw["format"].get("bit_rate", -1))
    else:
        info.duration = float(raw["streams"][0].get("duration", -1))
        info.bit_rate = float(raw["streams"][0].get("bit_rate", -1))
    info.frames = int(raw["streams"][0].get("nb_frames", -1))
    if info.frames == -1 and info.duration > 0 and info.frame_rate > 0:
        info.frames = int(info.duration * info.frame_rate)
    if len(raw["streams"]) > 1:
        info.audio_available = True
        info.audio_codec = raw["streams"][1].get("codec_name", None)
        info.audio_bitrate = float(raw["streams"][1].get("bit_rate", -1))
        info.audio_sample_rate = float(raw["streams"][1].get("sample_rate", -1))
        info.audio_channels = int(raw["streams"][1].get("channels", -1))
    return info, raw


def get_video_info(video_path: str) -> Tuple[float, float, int, int]:
    if False:  # use cv2
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoCapture.release()
        if fps == 0 or frames == 0 or width == 0 or height == 0:
            raise ValueError("Failed to get video info, may not a valid video file")
        return fps, frames, width, height
    else:  # use ffprobe
        info, _ = probe_video_info(video_path)
        if info is None:
            return 0, 0, 0, 0
        return (
            info.frame_rate,
            info.frames,
            info.width,
            info.height,
        )


def ffmpeg_merge_videos(vid_out: str, vid_in_list: List[str]) -> bool:
    list_file = os.path.join(os.path.dirname(vid_out), ".ffmpeg_merge_list")
    text = ""
    for file in vid_in_list:
        text += f"file '{file}'\n"
    logger.debug(f"Generated merge list:\n{text}")
    with open(list_file, "w", encoding="gbk", errors="ignore") as f:
        f.write(text)
    if os.path.exists(vid_out):
        os.remove(vid_out)
    with open(vid_out, "w") as _:
        pass
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", (list_file),
        "-c", "copy", vid_out,
    ]  # fmt: skip
    logger.debug(f"FFmpeg merging command: {command}")
    try:
        sp.check_output(command)
    except sp.CalledProcessError as e:
        logger.error(
            f'Failed to merge videos (return code: {e.returncode}), output:\n{e.output.decode("utf-8", "ignore")}'
        )
        return False
    os.remove(list_file)
    return True


def ffmpeg_merge_video_and_audio(
    vid_out: str,
    vid_in: str,
    audio_in: str,
) -> bool:
    if os.path.exists(vid_out):
        os.remove(vid_out)
    with open(vid_out, "w") as _:
        pass
    command = [
        "ffmpeg", "-y","-hide_banner", "-loglevel", "error",
        "-i", (vid_in), "-i", (audio_in),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        "-strict", "-2", vid_out,
    ]  # fmt: skip
    logger.debug(f"FFmpeg merging command: {command}")
    try:
        sp.check_output(command)
    except sp.CalledProcessError as e:
        logger.error(
            f'Failed to merge video and audio (return code: {e.returncode}), output:\n{e.output.decode("utf-8", "ignore")}'
        )
        return False
    return True


def find_unfinished_last_file(vid_out_name: str) -> Tuple[Optional[str], int]:
    search_name = os.path.basename(vid_out_name)
    search_name = os.path.splitext(search_name)[0]
    search_path = os.path.dirname(vid_out_name)
    file_list = os.listdir(search_path)
    max_to_num = -1
    max_to_file = None
    for file in file_list:
        if search_name in file:
            tmp = file.replace(search_name, "")
            to_text = re.search(r".*_to(\d+).*", tmp)
            if to_text:
                try:
                    to_num = int(to_text.group(1))
                    if to_num > max_to_num:
                        max_to_num = to_num
                        max_to_file = file
                except ValueError:
                    pass
    return max_to_file, max_to_num


def find_unfinished_merge_list(vid_out_name: str) -> Optional[List[str]]:
    search_name = os.path.basename(vid_out_name)
    search_name = os.path.splitext(search_name)[0]
    search_path = os.path.dirname(vid_out_name)
    merge_list = []
    file_list = os.listdir(search_path)
    start_frame = 0
    run = True
    ok = False
    while run:
        for file in file_list:
            if search_name in file:
                tmp = file.replace(search_name, "")
                if start_frame == 0:
                    found = re.search(r"^_to(\d+).*", tmp)
                    if found:
                        start_frame = int(found.group(1)) + 1
                        merge_list.append(os.path.join(search_path, file))
                        break
                else:
                    found = re.search(rf"^_from{start_frame}_to(\d+).*", tmp)
                    if found:
                        start_frame = int(found.group(1)) + 1
                        merge_list.append(os.path.join(search_path, file))
                        break
                    found = re.search(rf"^_from{start_frame}\..*", tmp)
                    if found:
                        merge_list.append(os.path.join(search_path, file))
                        ok = True
                        run = False
                        break
        else:
            logger.error(
                f"Failed to find file from {start_frame} (find {search_name} in {file_list})"
            )
            run = False
    if not ok:
        return None
    return merge_list


def check_ffmpeg_installed() -> None:
    try:
        out = sp.check_output(["ffmpeg", "-version"])
    except (sp.CalledProcessError, FileNotFoundError):
        logger.error(
            "FFmpeg not detected, please install it from https://ffmpeg.org/download.html and ADD TO PATH"
        )
        sys.exit(1)
    fd = re.search(r"ffmpeg version ([\w\-.]+)", out.decode("utf-8"))
    if fd:
        logger.success(f"FFmpeg detected (version: {fd.group(1)})")
    else:
        logger.warning("FFmpeg seems installed but version not detected")


def check_ffmepg_available_codec(
    find_codec=["h264", "hevc", "av1", "mpeg4"],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    return {codec_name: (encoders, decoders)}
    """
    try:
        out = sp.check_output(["ffmpeg", "-hide_banner", "-codecs"])
    except (sp.CalledProcessError, FileNotFoundError):
        logger.error(
            "FFmpeg not installed, please install it from https://ffmpeg.org/download.html and ADD TO PATH"
        )
        sys.exit(1)
    _codec_dict = {}
    for line in out.decode("utf-8", "ignore").split("\n"):
        try:
            enc = []
            dec = []
            codec = line.split(" ")[2]
            fd = re.search(r"\(decoders:([^()]+)\)", line)
            if fd:
                dec = fd.group(1).strip().split(" ")
            fe = re.search(r"\(encoders:([^()]+)\)", line)
            if fe:
                enc = fe.group(1).strip().split(" ")
            if codec in _codec_dict:
                enc = list(set(enc + _codec_dict[codec][0]))
                dec = list(set(dec + _codec_dict[codec][1]))
            if (not enc) and (not dec):
                continue
            _codec_dict[codec] = (enc, dec)
        except Exception:
            pass
    if find_codec is None:
        return _codec_dict
    codec_dict = {}  # sort by find_codec
    for codec in find_codec:
        if codec in _codec_dict:
            codec_dict[codec] = _codec_dict[codec]
    return codec_dict


if __name__ == "__main__":
    __import__("pprint").pprint(check_ffmepg_available_codec())
