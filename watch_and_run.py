import argparse
import subprocess
import time
import os
import sys
from pathlib import Path

# ANSI color/style constants for highlighting script output
RESET = "\033[0m"
CYAN = "\033[36;1;7m"  # Bold inverse cyan
RED = "\033[31;1;7m"  # Bold inverse red


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Watch a file and execute commands from a list when it changes.

        Substitution options for commands:
          %s - full path to the watched file
          %b - base filename without extension
          %e - extension (with dot, e.g. .bin)
          %d - directory of the watched file
          %h - home directory (OS style)
          %S - full path to the watched file (forward slashes)
          %D - directory of the watched file (forward slashes)
          %H - home directory (forward slashes)
          %t - timestamp in format YYYY_MM_DD-HH-MM-SS
        Example: python mmwave_to_soli.py --bin %s --out %b.h5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cmds", "-c", required=True,
        help="Text file with commands to run. Use %%s, %%b, %%e, %%d, %%S, %%D, %%h, %%H, %%t as placeholders for the watched file.")
    parser.add_argument("--dir", "-d", default=None, help="Directory to run commands in (default: current directory).")
    parser.add_argument("--override", "-o", action="store_true", help="Continue running commands even if one fails.")
    parser.add_argument("--interval", "-i", type=float, default=5.0, help="Polling interval in seconds (default: 5.0)")
    parser.add_argument("--run", "-r", action="store_true", help="Run commands immediately if the file exists at startup.")
    parser.add_argument("--once", "-1", action="store_true", help="Exit after running commands once on a change.")
    parser.add_argument("file", nargs=1, help="File to watch for changes.")
    args = parser.parse_args()
    # If nargs=1, flatten file argument
    if isinstance(args.file, list):
        args.file = args.file[0]
    return args


def read_commands(cmd_file):
    with open(cmd_file, 'r') as f:
        cmds = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return cmds


def run_commands(cmds, file_path, run_dir, override):
    file_path = Path(file_path)
    base = file_path.stem
    ext = file_path.suffix
    dir_ = str(file_path.parent)
    # Add forward-slash versions
    file_path_fwd = file_path.as_posix()
    dir_fwd = file_path.parent.as_posix()
    home_dir = str(Path.home())
    home_dir_fwd = Path.home().as_posix()
    script_dir = Path(__file__).parent.resolve()
    # Add timestamp substitution
    timestamp = time.strftime('%Y_%m_%d-%H-%M-%S')
    for cmd in cmds:
        cmd_to_run = cmd.replace('%s', str(file_path))
        cmd_to_run = cmd_to_run.replace('%b', base)
        cmd_to_run = cmd_to_run.replace('%e', ext)
        cmd_to_run = cmd_to_run.replace('%d', dir_)
        cmd_to_run = cmd_to_run.replace('%S', file_path_fwd)
        cmd_to_run = cmd_to_run.replace('%D', dir_fwd)
        cmd_to_run = cmd_to_run.replace('%h', home_dir)
        cmd_to_run = cmd_to_run.replace('%H', home_dir_fwd)
        cmd_to_run = cmd_to_run.replace('%t', timestamp)
        # If --dir is used, check if the command is a script in script_dir
        parts = cmd_to_run.strip().split()
        if parts:
            exe = parts[0]
            # Only check if not an absolute/relative path or in PATH
            if not (os.path.isabs(exe) or exe.startswith('.') or exe.startswith('/') or exe.startswith('\\')):
                # Check if exe exists in script_dir
                exe_path = script_dir / exe
                if exe_path.exists():
                    parts[0] = str(exe_path)
                    cmd_to_run = ' '.join(parts)
        print(f"{CYAN}[watch_and_run]{RESET} Running: {cmd_to_run}")
        try:
            #result = subprocess.run(cmd_to_run, shell=True, cwd=run_dir, executable="powershell.exe")
            result = subprocess.run(cmd_to_run, shell=True, cwd=run_dir)
            if result.returncode != 0:
                print(f"{RED}[watch_and_run]{RESET} Command failed with exit code {result.returncode}")
                if not override:
                    return False
        except Exception as e:
            print(f"{RED}[watch_and_run]{RESET} Error running command: {e}")
            if not override:
                return False
    return True


def main():
    args = parse_args()
    file_path = Path(args.file)
    cmds = read_commands(args.cmds)
    run_dir = args.dir if args.dir else os.getcwd()
    last_mtime = None
    print(f"{CYAN}[watch_and_run]{RESET} Watching {file_path} for changes. Running commands from {args.cmds} in {run_dir}.")
    ran_initial = False
    try:
        while True:
            # If --run is set and this is the first loop and the file exists, run immediately (no sleep)
            if last_mtime is None and args.run and not ran_initial and file_path.exists():
                mtime = file_path.stat().st_mtime
                print(f"{CYAN}[watch_and_run]{RESET} Change detected in {file_path} at {time.ctime(mtime)}.")
                success = run_commands(cmds, file_path, run_dir, args.override)
                last_mtime = mtime
                ran_initial = True
                if not success:
                    print(f"{RED}[watch_and_run]{RESET} Stopping due to command failure.")
                    break
                print(f"{CYAN}[watch_and_run]{RESET} Watching {file_path} for changes. Running commands from {args.cmds} in {run_dir}.")
                if args.once:
                    print(f"{CYAN}[watch_and_run]{RESET} Exiting after one run.")
                    break
                continue
            time.sleep(args.interval)
            if not file_path.exists():
                continue
            try:
                mtime = file_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if (last_mtime is None and args.run and not ran_initial) or (last_mtime is not None and mtime != last_mtime):
                print(f"{CYAN}[watch_and_run]{RESET} Change detected in {file_path} at {time.ctime(mtime)}.")
                success = run_commands(cmds, file_path, run_dir, args.override)
                last_mtime = mtime
                ran_initial = True
                if not success:
                    print(f"{RED}[watch_and_run]{RESET} Stopping due to command failure.")
                    break
                print(f"{CYAN}[watch_and_run]{RESET} Watching {file_path} for changes. Running commands from {args.cmds} in {run_dir}.")
                if args.once:
                    print(f"{CYAN}[watch_and_run]{RESET} Exiting after one run.")
                    break
            elif last_mtime is None:
                last_mtime = mtime
    except KeyboardInterrupt:
        print(f"\n{CYAN}[watch_and_run]{RESET} Stopped by user.")

if __name__ == "__main__":
    main()
