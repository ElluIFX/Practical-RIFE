import _thread
import os
import re
import subprocess as sp
import sys
import time
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skvideo.io
from loguru import logger
from tqdm import tqdm
from win32api import GetShortPathName as ShortName


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
        extra_opts: dict[str, str] = {},
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
        quality_option = "-crf" if "libx" in codec else "-q:v"
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", frame_fmt, "-r", str(fps),
            "-i", "-",
            "-c:v", codec, quality_option, str(quality)
        ]  # fmt: skip
        if extra_opts:
            for k, v in extra_opts.items():
                command.extend([k, v])
        command.append(ShortName(filename))
        logger.debug(f"FFmpeg backend command: {' '.join(command)}")
        self.proc = sp.Popen(
            command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, shell=False
        )
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
        extra_opts: dict[str, str] = {},
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
        inputdict: Optional[dict[str, str]] = None,
        buffer_size: int = 32,
    ) -> None:
        self.filename = filename
        if start_time > 0:
            _inputdict = {"-ss": str(start_time)}  # may not accurate
            if inputdict:
                _inputdict.update(inputdict)
            videogen = skvideo.io.vreader(ShortName(filename), inputdict=_inputdict)
            logger.info(f"Read video from {start_time}s")
        else:
            videogen = skvideo.io.vreader(ShortName(filename), inputdict=inputdict)
        if start_frame > 0:
            logger.info(f"Skipping {start_frame - 1} frames...")
            for _ in tqdm(range(start_frame - 1)):
                next(videogen)
            logger.success("Continue reading")
        self.videogen = videogen
        self.buffer = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.skip_frame = skip_frame
        self.running = True
        _thread.start_new_thread(self._worker, ())

    def close(self) -> None:
        self.running = False
        self.videogen.close()
        while not self.buffer.empty():
            self.buffer.get()

    def _worker(self):
        logger.success("Video reader started")
        try:
            if self.skip_frame == 0:
                for frame in self.videogen:
                    if not self.running:
                        break
                    self.buffer.put(frame)
            else:
                while self.running:
                    for _ in range(self.skip_frame):
                        next(self.videogen)
                    self.buffer.put(next(self.videogen))
        except StopIteration:
            pass
        except Exception as e:
            logger.exception(f"Error occurred while reading video: {e}")
        self.buffer.put(None)
        logger.success("Video reader closed")

    def get(self) -> np.ndarray:
        return self.buffer.get()

    @property
    def in_queue(self):
        return self.buffer.qsize()


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
        self._title = (
            "Frame preview (ESC: ON/OFF, SPACE: Switch source, W/S: Zoom, P: Stop)"
        )
        self._notify_callback = None
        self._show_empty = False
        self._empty_frame = np.zeros((180, 530, 3), dtype=np.uint8)
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

        self.add_source("All output frame", lambda _: True, default=True)
        writer.callback = self._show_frame

    def add_source(
        self, name: str, condition: Callable[[int], bool], default: bool = False
    ):
        self._sources[name] = condition
        if default:
            self._current_source = name

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
        sources = list(self._sources.keys())
        idx = sources.index(self._current_source)
        idx = (idx + 1) % len(sources)
        self._current_source = sources[idx]

    def _show_frame(self, frame: np.ndarray, frame_id: int) -> None:
        if frame is None or not self.running:
            return
        if frame_id == 0:
            cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
        if self._sources[self._current_source](frame_id):
            if self._notify_callback:
                extra_info, extra_notify = self._notify_callback(frame_id)
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
            text = f"Frame={int(frame_id):d}/{int(self._total_frame):d} {progress:.2%}"
            cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            ############### Write info text ###############
            y -= offset
            video_file_size = os.path.getsize(self._file_name) / 1024 / 1024
            frame_time = frame_id / self._fps
            total_time = self._total_frame / self._fps
            time_str = f"{int(frame_time / 60):02d}:{frame_time % 60:04.1f}/{int(total_time / 60):02d}:{int(total_time % 60):02d}"
            text = (
                f"Write:{video_file_size:.2f}MB {time_str}"
                if video_file_size < 1024
                else f"Write:{video_file_size/1024:.2f}GB {time_str}"
            )
            color = info_color
            if in_queue_write > 6:
                if in_queue_write >= buffer_size_write:
                    text += " [Write Quene Full]"
                else:
                    text += " [Write Slow]"
                color = warn_color
            if (
                in_queue_read < buffer_size_read - 6
                and self._total_frame - frame_id - in_queue_write > buffer_size_read
            ):
                text += " [Read Slow]"
                color = warn_color
            cv2.putText(temp, text, (x, y), font, scale, color, thickness)
            ############### Write extra info ###############
            if extra_info:
                for text in extra_info:
                    y -= offset
                    cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            ############### Write source text ###############
            y -= offset
            text = f"Source: {self._current_source}"
            cv2.putText(temp, text, (x, y), font, scale, info_color, thickness)
            ############### Write extra notify ###############
            if extra_notify:
                y -= offset
                cv2.putText(
                    temp, extra_notify, (x, y), font, scale, notify_color, thickness
                )
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


def ffmpeg_merge_videos(vid_out: str, vid_in_list: list[str]) -> bool:
    list_file = os.path.join(os.path.dirname(vid_out), ".merge_list.txt")
    text = ""
    for file in vid_in_list:
        text += f"file '{ShortName(file)}'\n"
    logger.debug(f"Generated merge list:\n{text}")
    with open(list_file, "w") as f:
        f.write(text)
    if os.path.exists(vid_out):
        os.remove(vid_out)
    with open(vid_out, "w") as _:
        pass
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", ShortName(list_file),
        "-c", "copy",
        ShortName(vid_out),
    ]  # fmt: skip
    logger.debug(f"FFmpeg merging command: {' '.join(command)}")
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
        "-i", ShortName(vid_in), "-i", ShortName(audio_in),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        ShortName(vid_out),
    ]  # fmt: skip
    logger.debug(f"FFmpeg merging command: {' '.join(command)}")
    try:
        sp.check_output(command)
    except sp.CalledProcessError as e:
        logger.error(
            f'Failed to merge video and audio (return code: {e.returncode}), output:\n{e.output.decode("utf-8", "ignore")}'
        )
        return False
    return True


def get_video_info(video_path: str) -> tuple[float, float, int, int]:
    videoCapture = cv2.VideoCapture(ShortName(video_path))
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoCapture.release()
    if fps == 0 or frames == 0 or width == 0 or height == 0:
        raise ValueError("Failed to get video info, may not a valid video file")
    return fps, frames, width, height


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


def find_unfinished_merge_list(vid_out_name: str) -> Optional[list[str]]:
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
            logger.error(f"Failed to find file from {start_frame}")
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
        logger.info("FFmpeg seems installed but version not detected")


# codec_name: (encoders, decoders)
def check_ffmepg_available_codec(
    find_codec=["h264", "hevc", "av1", "mpeg4"],
) -> Dict[str, Tuple[List[str], List[str]]]:
    try:
        out = sp.check_output(["ffmpeg", "-hide_banner", "-codecs"])
    except (sp.CalledProcessError, FileNotFoundError):
        logger.error(
            "FFmpeg not installed, please install it from https://ffmpeg.org/download.html"
        )
        sys.exit(1)
    _codec_dict = {}
    for line in out.decode("utf-8", "ignore").split("\n"):
        for codec in find_codec:
            if f" {codec} " in line:
                enc = []
                dec = []
                fd = re.search(r"\(decoders:([^()]+)\)", line)
                if fd:
                    dec = fd.group(1).strip().split(" ")
                fe = re.search(r"\(encoders:([^()]+)\)", line)
                if fe:
                    enc = fe.group(1).strip().split(" ")
                if codec in _codec_dict:
                    enc = list(set(enc + _codec_dict[codec][0]))
                    dec = list(set(dec + _codec_dict[codec][1]))
                _codec_dict[codec] = (enc, dec)
    codec_dict = {}  # sort by find_codec
    for codec in find_codec:
        if codec in _codec_dict:
            codec_dict[codec] = _codec_dict[codec]
    return codec_dict


if __name__ == "__main__":
    print(check_ffmepg_available_codec())
