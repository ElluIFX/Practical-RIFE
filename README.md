# Practical-RIFE

> [!IMPORTANT]
> **This fork is focused on improving the user experience when inference video**

Use `batch_inference.py` to interpolate multiple videos at once.

Or check `inference_video.py -h` for more options.

> [!NOTE]
> Current model version: V4.17

## Basic Usage

```bash
# Install dependencies (notice that this will install torch with CUDA support, better run in venv or conda)
python3 -m pip install -r requirements.txt

# Interpolate a video
python3 inference_video.py /path/to/video.mp4
```

## Special Arguments Explanation

- `--ssim`: Program use SSIM to detect scene change, this argument can change the threshold of SSIM, default is `0.4`, higher value means more sensitive to scene change, for a video with rapid change, you may want to lower this value (even set to `0`) for less mistaking scene change detection.
- `--scene_copy`: By default, if the scene change is detected, the model will stack the previous frame with the current frame to make the transition smoother (both 50% opacity). This argument will change this behavior to simply copy the previous frame as transition frame.
- `--start_frame`: Start frame for inference, program will automatically detect unfinished work (stopped by hit `P` key in preview window) and continue from there, so you dont need to specify this argument in normal cases.
- `--start_time` and `--stop_time`: Does same thing as `--start_frame`, but with time in seconds. also useless in normal cases.
- `--skip_frame`: Skip frame for inference, useful in case you want to re-interpolate a video with a new model version, but you have already deleted the original low-fps file, set `skip_frame` to `1` (or `2` for 3x interpolation, etc) to just read the raw frames from a interpolated video.
- `--headless`: Run the program without preview window, useful for running on server or headless machine.

## Screenshots

![1714128717467](image/README/1714128717467.png)

![1714128731690](image/README/1714128731690.png)
