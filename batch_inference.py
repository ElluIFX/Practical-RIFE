import os
import sys
import time

path = os.path.abspath(os.path.dirname(__file__))
target = r"./inference_video.py"

filelist = []
arg_multi = []
arg_UHD = []

if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print("Changed cwd to: " + path)

path = r'"' + path + r'"'


def get_arg():
    get = input("> FPS multi (2): ").strip()
    if get == "":
        get = "2"
    get = int(get)
    arg_multi.append(get)

    get = input("> Is UHD (y/N): ").strip()
    if "y" in get.lower():
        arg_UHD.append(True)
    else:
        arg_UHD.append(False)
    print()


i = 0
in_file = sys.argv[1:]
for f in in_file:
    i += 1
    if not f.startswith(r'"'):
        f = r'"' + f + r'"'
    filelist.append(f)
    print(f"Get file-{i:02d} path: ", f)

    get_arg()

while True:
    i += 1
    get = input(f"Input file-{i:02d} path: ").strip()
    if get == "":
        break
    if not get.startswith(r'"'):
        get = r'"' + get + r'"'
    filelist.append(get)

    get_arg()

poweroff = False
crf = 17
ssim = 0.4
fp16 = False
extra_args = ""
get = input("> Poweroff after inference (y/N): ").strip()
if "y" in get.lower():
    poweroff = True
get = input(f"> CRF ({crf}): ").strip()
if get != "":
    crf = int(get)
get = input(f"> SSIM ({ssim}): ").strip()
if get != "":
    ssim = float(get)
extra_args += f"--crf {crf} --ssim {ssim} "
get = input("> FP16 (y/N): ").strip()
if "y" in get.lower():
    fp16 = True
    extra_args += "--fp16 "
# get = input("> HEVC (y/N): ").strip()
# if "y" in get.lower():
#     extra_args += "--encoder hevc_qsv "


def get_extra_args():
    global extra_args
    get = input("> Extra args (? for help): ").strip()
    if len(get) == 0:
        return
    if get[0] == "?":
        command = f"py {target} --help"
        os.system(command)
        return get_extra_args()
    elif get != "":
        extra_args += get


get_extra_args()

i = 0
for file, multi, UHD in zip(filelist, arg_multi, arg_UHD):
    i += 1
    print(f"File-{i:02d}: {file}\n> Args: fps-multi={multi} Is-UHD={UHD}\n")
print(f"Global args: poweroff={poweroff} extra_args={extra_args}")

print()
input("Check above. Press Enter to continue...")
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
input("Press Enter to exit")
