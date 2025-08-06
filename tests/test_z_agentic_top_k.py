import sys
import os
import asyncio
from pathlib import Path
import sys
import os
import asyncio
import types
from pathlib import Path
import importlib

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DISCORD_BOT_TOKEN", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ["RERANKING_CANDIDATES"] = "20"

# Minimal discord stub for importing bot module
if 'discord' not in sys.modules:
    discord_mod = types.ModuleType('discord')
    sys.modules['discord'] = discord_mod
else:
    discord_mod = sys.modules['discord']

class Intents:
    def __init__(self):
        self.message_content = False

    @staticmethod
    def default():
        return Intents()

discord_mod.Intents = Intents
discord_mod.Interaction = getattr(discord_mod, 'Interaction', object)

class _Empty:
    pass

discord_mod.TextChannel = _Empty
discord_mod.Message = _Empty
discord_mod.Member = _Empty

app_cmds = types.ModuleType('discord.app_commands')
class DummyCheckFailure(Exception):
    pass
app_cmds.CheckFailure = DummyCheckFailure
discord_mod.app_commands = app_cmds

ext_module = types.ModuleType('discord.ext')
commands_module = types.ModuleType('discord.ext.commands')
commands_module.Bot = type('DummyBot', (), {})
commands_module.Context = _Empty
tasks_module = types.ModuleType('discord.ext.tasks')
def loop(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
tasks_module.loop = loop
ext_module.commands = commands_module
ext_module.tasks = tasks_module
sys.modules['discord.app_commands'] = app_cmds
sys.modules['discord.ext'] = ext_module
sys.modules['discord.ext.commands'] = commands_module
sys.modules['discord.ext.tasks'] = tasks_module

bot_module = importlib.import_module("bot")


class DummyConfig:
    TOP_K = 5
    MAX_TOP_K = 10
    KEYWORD_DATABASE_ENABLED = False

    def get_top_k_for_model(self, model: str) -> int:
        return self.MAX_TOP_K


class StubDocumentManager:
    async def search(self, query, top_k):
        return [
            ("doc", "Title", f"chunk{i}", 0.1, None, i + 1, top_k)
            for i in range(top_k)
        ]

    def search_keyword(self, keyword, top_k):
        return [
            ("doc", "Title", f"chunk{i}", i + 1, top_k)
            for i in range(top_k)
        ]

    def search_keyword_bm25(self, keyword, top_k):
        return [
            ("doc", "Title", f"chunk{i}", i + 1, top_k)
            for i in range(top_k)
        ]


def test_tool_search_documents_uses_requested_top_k():
    config = DummyConfig()
    dm = StubDocumentManager()
    bot = types.SimpleNamespace(
        document_manager=dm,
        config=config,
        _agentic_top_k_limit=config.get_top_k_for_model("model"),
    )
    results = asyncio.run(bot_module.DiscordBot._tool_search_documents(bot, "q", top_k=8))
    assert len(results) == 8


def test_tool_search_documents_clamps_to_model_limit():
    config = DummyConfig()
    dm = StubDocumentManager()
    bot = types.SimpleNamespace(
        document_manager=dm,
        config=config,
        _agentic_top_k_limit=config.get_top_k_for_model("model"),
    )
    results = asyncio.run(bot_module.DiscordBot._tool_search_documents(bot, "q", top_k=15))
    assert len(results) == config.MAX_TOP_K


def test_tool_search_keyword_allows_custom_top_k():
    config = DummyConfig()
    dm = StubDocumentManager()
    bot = types.SimpleNamespace(
        document_manager=dm,
        config=config,
        _agentic_top_k_limit=config.get_top_k_for_model("model"),
    )
    results = asyncio.run(bot_module.DiscordBot._tool_search_keyword(bot, keyword="foo", top_k=7))
    assert len(results) == 7

