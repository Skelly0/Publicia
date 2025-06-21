import os
import importlib.util
from pathlib import Path
import sys
import types
import pytest


def load_helpers_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "helpers.py"

    # Stub discord and config dependencies to avoid heavy imports
    if 'discord' not in sys.modules:
        dummy_discord = types.ModuleType('discord')
        dummy_discord.Interaction = object
        app_cmds = types.ModuleType('discord.app_commands')
        class DummyCheckFailure(Exception):
            pass
        app_cmds.CheckFailure = DummyCheckFailure
        dummy_discord.app_commands = app_cmds
        sys.modules['discord'] = dummy_discord
        sys.modules['discord.app_commands'] = app_cmds

    if 'managers.config' not in sys.modules:
        config_mod = types.ModuleType('managers.config')
        class DummyConfig:
            ALLOWED_USER_IDS = []
            ALLOWED_ROLE_IDS = []
            def __init__(self):
                self.DISCORD_BOT_TOKEN = 'x'
                self.OPENROUTER_API_KEY = 'x'
                self.GOOGLE_API_KEY = 'x'
            def _validate_config(self):
                pass
        config_mod.Config = DummyConfig
        sys.modules['managers.config'] = config_mod

    spec = importlib.util.spec_from_file_location('helpers', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_helpers_module()


def test_sanitize_filename_basic():
    assert helpers.sanitize_filename('test<invalid>.txt') == 'test_invalid_.txt'


def test_sanitize_filename_reserved():
    assert helpers.sanitize_filename('CON') == '_CON'


def test_split_message_simple():
    text = 'hello world'
    assert helpers.split_message(text, max_length=20) == ['hello world']


def test_split_message_long():
    text = ('a' * 30 + '\n\n') * 10
    chunks = helpers.split_message(text, max_length=50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert ''.join(chunk.replace('\n', '') for chunk in chunks) .startswith('a' * 30)
