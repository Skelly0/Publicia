import importlib.util
from pathlib import Path

import pytest


def load_preferences_module():
    module_path = Path(__file__).resolve().parents[1] / "managers" / "preferences.py"
    spec = importlib.util.spec_from_file_location("preferences", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_toggle_debug_mode_with_faulty_path(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    # Inject an invalid preferences directory to trigger an early failure
    manager.preferences_dir = None
    result = manager.toggle_debug_mode("user")
    assert result is False
