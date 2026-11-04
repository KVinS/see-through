"""Apply the Sugoi UI part of the Deleted feature to ../tools, once.

The desktop wrapper lives outside this Git checkout. Keep its small feature diff
here so a single git format-patch includes both the inference and UI changes.
"""
from pathlib import Path
import shutil
import subprocess


def main():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    if not (root / 'tools' / 'sugoi_gui.py').is_file():
        raise SystemExit('Sugoi installation not found: expected ../../tools/sugoi_gui.py')
    local_git = root / '.runtime' / 'git' / 'cmd' / 'git.exe'
    git = str(local_git) if local_git.is_file() else shutil.which('git')
    if git is None:
        raise SystemExit('Git not found. Run from an installed Sugoi environment.')
    command = [git, '-C', str(root), 'apply']
    patch = str(package / 'save_deleted_ui.patch')

    def check(*args):
        return subprocess.run(command + list(args) + [patch], capture_output=True,
                              encoding='utf-8', errors='replace')

    if check('--reverse', '--check').returncode == 0:
        print('Save Deleted UI is already installed.')
        return
    result = check('--check')
    if result.returncode:
        raise SystemExit('UI files differ from the supported version; no files changed.\n' + result.stderr)
    subprocess.run(command + [patch], check=True)
    print('Save Deleted UI installed. Restart Open_UI.bat.')


if __name__ == '__main__':
    main()
