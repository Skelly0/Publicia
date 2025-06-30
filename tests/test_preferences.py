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


def test_toggle_info_prompt_with_faulty_path(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    manager.preferences_dir = None
    result = manager.toggle_informational_prompt_mode("user")
    assert result is False


def test_set_temperature_with_faulty_path(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    manager.preferences_dir = None
    result = manager.set_temperature_settings("user", 0.0, 0.1, 0.4)
    assert result is False


def test_set_temperature_invalid_range(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    # base higher than max should fail
    result = manager.set_temperature_settings("user", 0.0, 0.8, 0.6)
    assert result is False


def test_set_temperature_valid_range(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    result = manager.set_temperature_settings("user", 0.1, 0.2, 0.5)
    assert result is True
    temps = manager.get_temperature_settings("user")
    assert temps == (0.1, 0.2, 0.5)


def test_clear_temperature_settings(tmp_path):
    prefs = load_preferences_module()
    manager = prefs.UserPreferencesManager(base_dir=str(tmp_path))
    manager.set_temperature_settings("user", 0.2, 0.3, 0.6)
    cleared = manager.clear_temperature_settings("user")
    assert cleared is True
    temps = manager.get_temperature_settings("user")
    assert temps == (None, None, None)
