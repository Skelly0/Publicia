import json
import os
import importlib.util
from pathlib import Path


def load_conversation_module():
    module_path = Path(__file__).resolve().parents[1] / "managers" / "conversation.py"
    spec = importlib.util.spec_from_file_location("conversation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conversation = load_conversation_module()
ConversationManager = conversation.ConversationManager


def test_get_file_path_sanitization(tmp_path):
    manager = ConversationManager(base_dir=str(tmp_path))
    path = manager.get_file_path('User/../bad')
    assert os.path.basename(path) == 'User..bad.json'


def test_write_and_get_limited_history(tmp_path):
    manager = ConversationManager(base_dir=str(tmp_path))
    for i in range(5):
        manager.write_conversation('user', 'user', f'msg {i}')
    history = manager.get_limited_history('user', limit=3)
    assert len(history) == 3
    assert history[0]['display_index'] == 0
    assert history[-1]['content'].endswith('msg 4')


def test_delete_messages_by_display_index(tmp_path):
    manager = ConversationManager(base_dir=str(tmp_path))
    for i in range(5):
        manager.write_conversation('user', 'user', f'msg {i}')
    success, _, count = manager.delete_messages_by_display_index('user', [1], limit=3)
    assert success is True
    with open(manager.get_file_path('user'), 'r', encoding='utf-8') as f:
        messages = json.load(f)
    assert len(messages) == 4
