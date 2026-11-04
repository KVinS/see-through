"""Sugoi wrapper integration. Run from repo: python -m unittest discover -s sugoi -v."""
import configparser
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
TOOLS = PACKAGE.parent.parent / 'tools'
sys.path.insert(0, str(TOOLS))


@unittest.skipUnless((TOOLS / 'sugoi_gui.py').is_file(), 'Sugoi wrapper is not installed')
class SaveDeletedUITests(unittest.TestCase):
    def test_checkbox_persists_and_reaches_both_command_builders(self):
        import benchmark_cold
        import run_queue
        import sugoi_gui
        import sugoi_settings

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sugoi_settings, 'CONFIG', Path(tmp) / 'config.ini'), \
                patch.object(sugoi_gui, 'TaskbarProgress'), \
                patch.object(sugoi_gui, 'load_settings', return_value=sugoi_settings.DEFAULTS.copy()):
            config = sugoi_settings.CONFIG
            config.write_text('[queue]\npause_minutes=10\nskip_existing=true\ncontinue_on_error=true\n', encoding='utf-8')
            window = tk.Tk()
            window.withdraw()
            app = sugoi_gui.SugoiUI(window)
            try:
                self.assertFalse(app.save_deleted.get())
                self.assertEqual(app.deleted_check.cget('text'), 'Сохранять удалённое')
                app.deleted_check.invoke()
                sugoi_settings.save_settings(app.get_values())
                loaded = sugoi_settings.load_settings()
                self.assertTrue(loaded['save_deleted'])
                app.set_values(loaded)
                self.assertTrue(app.save_deleted.get())
                queue_settings = run_queue.load_config(config)
                for settings, builder in [(loaded, benchmark_cold.build_command),
                                          (queue_settings, run_queue.build_command)]:
                    args = (Path('image.png'), settings, Path(tmp)) if builder == benchmark_cold.build_command else (Path('image.png'), settings)
                    self.assertIn('--save_deleted', builder(*args))
                    settings['save_deleted'] = False
                    self.assertNotIn('--save_deleted', builder(*args))
                cfg = configparser.ConfigParser()
                cfg.read(config)
                self.assertEqual(cfg.getint('queue', 'pause_minutes'), 10)
                # Existing installations need no config.ini migration.
                cfg.remove_option('processing', 'save_deleted')
                with config.open('w', encoding='utf-8') as file:
                    cfg.write(file)
                self.assertFalse(sugoi_settings.load_settings()['save_deleted'])
                self.assertFalse(run_queue.load_config(config)['save_deleted'])
            finally:
                app.destroy()

    def test_packaged_ui_patch_installs_and_is_idempotent(self):
        git = shutil.which('git')
        if git is None:
            self.skipTest('Git unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'repo' / 'sugoi'
            package.mkdir(parents=True)
            (root / 'tools').mkdir()
            for name in ['install_save_deleted.py', 'save_deleted_ui.patch']:
                shutil.copyfile(PACKAGE / name, package / name)
            names = ['sugoi_gui.py', 'sugoi_settings.py', 'benchmark_cold.py', 'run_queue.py']
            for name in names:
                shutil.copyfile(TOOLS / name, root / 'tools' / name)
            subprocess.run([git, '-C', str(root), 'apply', '--reverse', str(package / 'save_deleted_ui.patch')], check=True, capture_output=True)
            for message in ['installed.', 'already installed.']:
                result = subprocess.run([sys.executable, str(package / 'install_save_deleted.py')], check=True, capture_output=True, text=True)
                self.assertIn(message, result.stdout)
            for name in names:
                self.assertEqual((root / 'tools' / name).read_text(encoding='utf-8'),
                                 (TOOLS / name).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
