
import os
import sys
import subprocess

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    ui_folder    = os.path.join(project_root, "UI")
    main_script  = os.path.join(ui_folder, "main.py")

    if not os.path.isfile(main_script):
        print(f"Error: {main_script!r} not found.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    # ensure Python runs in UTF-8 mode
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    args = [sys.executable, main_script] + sys.argv[1:]

    try:
        subprocess.run(args, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
