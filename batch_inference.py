import os
import sys
import time

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from utils import get_video_info

path = os.path.abspath(os.path.dirname(__file__))
target = r"./inference_video.py"

filelist = []
arg_multi = []
arg_UHD = []

if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print("Changed cwd to: " + path)

path = r'"' + path + r'"'


def get_arg(file):
    file = os.path.abspath(file)
    fps, tot_frame, width, height = get_video_info(file)
    print(f"Info: {fps:.2f} fps | {tot_frame} frames | {width}x{height}")
    fps = IntPrompt.ask("FPS multi", default=2)
    arg_multi.append(fps)
    uhd = Confirm.ask("Is UHD", default=False)
    arg_UHD.append(uhd)
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
    print(f"Get file-{i:02d} path: ", ff)
    get_arg(f)

while True:
    i += 1
    get = Prompt.ask(f"Input file-{i:02d} path").strip()
    if get == "":
        break
    if not get.startswith(r'"'):
        ff = r'"' + f + r'"'
    else:
        ff = get
    filelist.append(ff)
    get_arg(get)

extra_args = ""
poweroff = Confirm.ask("Poweroff after inference", default=False)
crf = IntPrompt.ask("Quality", default=17)
ssim = FloatPrompt.ask("SSIM", default=0.4)
extra_args += f"--quality {crf} --ssim {ssim} "
fp16 = Confirm.ask("Use FP16", default=False)
if fp16:
    extra_args += "--fp16 "
if Confirm.ask("Use HEVC", default=False):
    extra_args += "--codec hevc_qsv "


def get_extra_args():
    get = Prompt.ask("> Extra args (? for help)").strip()
    if len(get) == 0:
        return ""
    if get[0] == "?":
        command = f"py {target} --help"
        os.system(command)
        return get_extra_args()
    elif get != "":
        return " " + get
    return ""


extra_args += get_extra_args()

i = 0
for file, multi, UHD in zip(filelist, arg_multi, arg_UHD):
    i += 1
    print(f"File-{i:02d}: {file}\n> Args: fps-multi={multi} Is-UHD={UHD}\n")
print(f"Global args: poweroff={poweroff} extra_args={extra_args}")

print()
if not Confirm.ask("Check above, confirm to start", default=True):
    print("Canceled")
    sys.exit(0)
print("Starting inference")

error_files = []
t_start = time.time()
i = 0
for file, multi, UHD in zip(filelist, arg_multi, arg_UHD):
    i += 1
    command = f"py {target} --multi {multi} "
    if UHD:
        command += "--UHD "
    command += f"{extra_args} "
    command += file
    print(f"[{i}/{len(filelist)}] command: {command}")
    ret = os.system(command)
    print(f"[{i}/{len(filelist)}] Done file: {file} ret: {ret}")
    if ret != 0:
        error_files.append(file)

print("Finished inference")
print(f"Total cost: {(time.time() - t_start)/60:.2f} mins")
print("Some files returned error:", error_files)
if poweroff:
    print("Poweroff in 60 seconds")
    os.system("shutdown -s -t 60")
Prompt.ask("Press Enter to exit")
