import importlib.util
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "managers" / "doc_tracking_channels.py"
    spec = importlib.util.spec_from_file_location("doc_channels", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_and_remove_channel(tmp_path):
    mod = load_module()
    manager = mod.DocTrackingChannelManager(base_dir=str(tmp_path))
    assert manager.get_channels() == []

    added = manager.add_channel(123)
    assert added is True
    assert manager.get_channels() == [123]

    added_again = manager.add_channel(123)
    assert added_again is False
    assert manager.get_channels() == [123]

    removed = manager.remove_channel(123)
    assert removed is True
    assert manager.get_channels() == []
