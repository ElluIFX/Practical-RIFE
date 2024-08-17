import os
import sys
import time

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from trained import DEFAULT_MODEL, MODEL_LIST
from utils import get_video_info

path = os.path.abspath(os.path.dirname(__file__))
target = f"{sys.executable} ./inference_video.py"
filelist = []
arg_multi = []
arg_uhd = []

print = Console().print
if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print(f"[blue]Changed cwd to: {path}")

path = r'"' + path + r'"'


def get_arg(file):
    file = os.path.abspath(file)
    fps, tot_frame, width, height = get_video_info(file)
    print(f"[blue]Info: {fps:.2f} fps | {tot_frame} frames | {width}x{height}")
    fps = IntPrompt.ask("[green]FPS multi", default=2)
    arg_multi.append(fps)
    uhd = Confirm.ask(
        "[green]High-Res source", default=width * height > 1920 * 1080 * 1.2
    )
    arg_uhd.append(uhd)
    print()


i = 0
in_file = sys.argv[1:]
for f in in_file:
    i += 1
    if not f.startswith(r'"'):
        ff = r'"' + f + r'"'
    else:
        ff = f
    filelist.append(ff)
    print(f"[green]Get file-{i:02d} path: [reset] {f}")
    get_arg(f)

while True:
    i += 1
    get = Prompt.ask(
        f"[green]Input file-{i:02d} path [yellow](return to finish)"
    ).strip()
    if get == "":
        break
    if not get.startswith(r'"'):
        ff = r'"' + get + r'"'
    else:
        ff = get
    filelist.append(ff)
    get_arg(get)

extra_args = ""
poweroff = Confirm.ask("[green]Poweroff after inference", default=False)
crf = IntPrompt.ask("[green]Quality", default=17)
ssim = FloatPrompt.ask("[green]SSIM", default=0.4)
extra_args += f"--quality {crf} --ssim {ssim} "
model = Prompt.ask("[green]Model", choices=MODEL_LIST, default=DEFAULT_MODEL)
extra_args += f"--model {model} "
if Confirm.ask("[green]Use FP16 if possible", default=True):
    extra_args += "--fp16 "
codec = Prompt.ask("[green]Codec", default="h264_nvenc")
extra_args += f"--codec {codec} "


def get_extra_args():
    get = Prompt.ask("[green]> Extra args [yellow](? for help)").strip()
    if len(get) == 0:
        return ""
    if get[0] == "?":
        command = f"{target} --help"
        os.system(command)
        return get_extra_args()
    elif get != "":
        return " " + get
    return ""


extra_args += get_extra_args()

i = 0
for file, multi, uhd in zip(filelist, arg_multi, arg_uhd):
    i += 1
    print(f"[blue]File-{i:02d}: {file}\n[green]> Args: --multi={multi} --uhd={uhd}\n")
print(f"[green]Global args: poweroff={poweroff} extra_args={extra_args}")

print()
if not Confirm.ask("[yellow]Check above, confirm to start", default=True):
    print("[red]Canceled")
    sys.exit(0)
print("[green]Starting inference")

error_files = []
t_start = time.time()
i = 0
for file, multi, uhd in zip(filelist, arg_multi, arg_uhd):
    i += 1
    command = f"{target} --multi {multi} "
    if uhd:
        command += "--uhd "
    command += f"{extra_args} "
    command += file
    print(f"[yellow][{i}/{len(filelist)}] command: {command}")
    ret = os.system(command)
    print(f"[green][{i}/{len(filelist)}] Done file: {file} ret: {ret}")
    if ret != 0:
        error_files.append(file)

print("[green]Finished inference")
print(f"[yellow]Total cost: {(time.time() - t_start)/60:.2f} mins")
if error_files:
    print("[red]Some files returned error:\n" + "\n".join(error_files))
if poweroff:
    print("[yellow]Poweroff in 60 seconds")
    os.system("shutdown -s -t 60")
print("[yellow]Press Enter to exit")
input()
