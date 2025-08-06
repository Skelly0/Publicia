import importlib.util
from pathlib import Path
import sys
import types
import importlib.util
import json
import pytest
import asyncio


def load_document_module():
    module_path = Path(__file__).resolve().parents[1] / "managers" / "documents.py"
    # Stub google.generativeai to avoid heavy import and API calls
    if 'google.generativeai' not in sys.modules:
        google_pkg = types.ModuleType('google')
        genai_mod = types.ModuleType('google.generativeai')
        def configure(api_key=None):
            pass
        genai_mod.configure = configure
        types_mod = types.ModuleType('google.generativeai.types')
        class HarmCategory:
            pass
        class HarmBlockThreshold:
            pass
        types_mod.HarmCategory = HarmCategory
        types_mod.HarmBlockThreshold = HarmBlockThreshold
        google_pkg.generativeai = genai_mod
        sys.modules['google'] = google_pkg
        sys.modules['google.generativeai'] = genai_mod
        sys.modules['google.generativeai.types'] = types_mod
    if 'numpy' not in sys.modules:
        numpy_dummy = types.ModuleType('numpy')
        numpy_dummy.ndarray = object
        def array(val):
            return val
        numpy_dummy.array = array
        sys.modules['numpy'] = numpy_dummy
    if 'aiohttp' not in sys.modules:
        aiohttp_dummy = types.ModuleType('aiohttp')
        sys.modules['aiohttp'] = aiohttp_dummy
    if 'rank_bm25' not in sys.modules:
        bm25_dummy = types.ModuleType('rank_bm25')
        class DummyBM25:
            def __init__(self, corpus, *args, **kwargs):
                self.corpus = corpus

            def get_scores(self, query_tokens):
                scores = []
                for doc in self.corpus:
                    # simple match score: count occurrences of any token
                    score = sum(1 for token in query_tokens if token in doc)
                    scores.append(float(score))
                return scores

        bm25_dummy.BM25Okapi = DummyBM25
        sys.modules['rank_bm25'] = bm25_dummy
    if 'utils.logging' not in sys.modules:
        utils_pkg = types.ModuleType('utils')
        logging_pkg = types.ModuleType('utils.logging')
        def sanitize_for_logging(message):
            return message
        logging_pkg.sanitize_for_logging = sanitize_for_logging
        helpers_pkg = types.ModuleType('utils.helpers')
        def dummy(*args, **kwargs):
            return None
        logging_pkg.log_qa_pair = dummy
        helpers_pkg.check_permissions = dummy
        helpers_pkg.is_image = lambda *a, **kw: False
        helpers_pkg.split_message = lambda msg, limit=2000: [msg]
        helpers_pkg.sanitize_filename = lambda name: name
        helpers_pkg.xml_wrap = lambda tag, content: f"<{tag}>{content}</{tag}>"
        helpers_pkg.wrap_document = lambda content, source, metadata="": content
        utils_pkg.logging = logging_pkg
        utils_pkg.helpers = helpers_pkg
        sys.modules['utils'] = utils_pkg
        sys.modules['utils.logging'] = logging_pkg
        sys.modules['utils.helpers'] = helpers_pkg
    if 'dotenv' not in sys.modules:
        dotenv_dummy = types.ModuleType('dotenv')
        def load_dotenv():
            pass
        dotenv_dummy.load_dotenv = load_dotenv
        sys.modules['dotenv'] = dotenv_dummy
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
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("documents", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bot_module():
    module_path = Path(__file__).resolve().parents[1] / "bot.py"

    # Ensure discord stubs with required attributes
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
    # Basic discord classes used in type hints
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
    class DummyBot:
        def __init__(self, *args, **kwargs):
            pass
    commands_module.Bot = DummyBot
    class Context:
        pass
    commands_module.Context = Context
    tasks_module = types.ModuleType('discord.ext.tasks')
    def loop(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    tasks_module.loop = loop
    ext_module.commands = commands_module
    ext_module.tasks = tasks_module
    discord_mod.ext = ext_module
    sys.modules['discord.app_commands'] = app_cmds
    sys.modules['discord.ext'] = ext_module
    sys.modules['discord.ext.commands'] = commands_module
    sys.modules['discord.ext.tasks'] = tasks_module

    # Stub commands package to avoid heavy imports
    if 'commands' not in sys.modules:
        commands_pkg = types.ModuleType('commands')
        for name in [
            'document_commands',
            'image_commands',
            'conversation_commands',
            'admin_commands',
            'utility_commands',
            'query_commands',
            'tracking_commands',
        ]:
            sub = types.ModuleType(f'commands.{name}')
            setattr(commands_pkg, name, sub)
            sys.modules[f'commands.{name}'] = sub
        sys.modules['commands'] = commands_pkg

    # Stub apscheduler
    if 'apscheduler.schedulers.asyncio' not in sys.modules:
        apscheduler_pkg = types.ModuleType('apscheduler')
        schedulers_pkg = types.ModuleType('apscheduler.schedulers')
        asyncio_pkg = types.ModuleType('apscheduler.schedulers.asyncio')
        class DummyScheduler:
            def add_job(self, *a, **kw):
                pass

            def start(self):
                pass
        asyncio_pkg.AsyncIOScheduler = DummyScheduler
        schedulers_pkg.asyncio = asyncio_pkg
        apscheduler_pkg.schedulers = schedulers_pkg
        sys.modules['apscheduler'] = apscheduler_pkg
        sys.modules['apscheduler.schedulers'] = schedulers_pkg
        sys.modules['apscheduler.schedulers.asyncio'] = asyncio_pkg

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("bot", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


documents = load_document_module()
DocumentManager = documents.DocumentManager


class DummyConfig:
    DOCUMENT_LIST_ENABLED = False
    GOOGLE_API_KEY = 'x'
    EMBEDDING_MODEL = 'models/text-embedding-004'
    EMBEDDING_DIMENSIONS = 0
    def __init__(self):
        pass


def test_document_list_disabled(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    # Ensure no metadata so basic list would be generated if enabled
    manager.metadata = {}
    assert manager.get_document_list_content() == ""


def test_search_keyword(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.chunks = {"abc": ["An airship sails the skies", "Nothing here"]}
    manager.metadata = {"abc": {"original_name": "TestDoc"}}

    results = manager.search_keyword("airship")
    assert len(results) == 1
    doc_uuid, name, chunk, index, total = results[0]
    assert doc_uuid == "abc"
    assert name == "TestDoc"
    assert index == 1 and total == 2
    assert "airship" in chunk.lower()


def test_search_keyword_bm25(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.chunks = {"abc": ["An airship sails the skies", "Nothing here"]}
    manager.metadata = {"abc": {"original_name": "TestDoc"}}

    # BM25 search should also find the chunk containing the keyword
    results = manager.search_keyword_bm25("airship")
    assert len(results) == 1
    doc_uuid, name, chunk, index, total = results[0]
    assert doc_uuid == "abc"
    assert name == "TestDoc"
    assert index == 1 and total == 2
    assert "airship" in chunk.lower()


def test_view_chunks_accepts_identifiers(tmp_path):
    bot_mod = load_bot_module()
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.chunks = {"uuid-123": ["First chunk", "Second"]}
    manager.contextualized_chunks = {"uuid-123": ["Ctx first", "Ctx second"]}
    manager.metadata = {"uuid-123": {"original_name": "MyDoc"}}
    tracked = [
        {
            "google_doc_id": "gdoc123",
            "internal_doc_uuid": "uuid-123",
            "original_name_at_import": "MyDoc",
        }
    ]
    (Path(tmp_path) / "tracked_google_docs.json").write_text(json.dumps(tracked))

    dummy_bot = types.SimpleNamespace(
        document_manager=manager,
        config=types.SimpleNamespace(USE_CONTEXTUALISED_CHUNKS=True),
    )

    res_uuid = asyncio.run(bot_mod.DiscordBot._tool_view_chunks(dummy_bot, "uuid-123", [1]))
    res_name = asyncio.run(bot_mod.DiscordBot._tool_view_chunks(dummy_bot, "MyDoc", [1]))
    res_gdoc = asyncio.run(bot_mod.DiscordBot._tool_view_chunks(dummy_bot, "gdoc123", [1]))
    res_gdoc_url = asyncio.run(
        bot_mod.DiscordBot._tool_view_chunks(
            dummy_bot, "https://docs.google.com/document/d/gdoc123/edit", [1]
        )
    )

    assert res_uuid == res_name == res_gdoc == res_gdoc_url
    assert res_uuid[0]["content"] == "First chunk"


def test_get_document_summary_returns_existing(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.metadata = {"uuid-123": {"original_name": "MyDoc", "summary": "Existing"}}
    summary = asyncio.run(manager.get_document_summary("MyDoc"))
    assert summary == "Existing"


def test_get_document_summary_generates_if_missing(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.metadata = {"uuid-123": {"original_name": "MyDoc"}}
    (Path(tmp_path) / "uuid-123.txt").write_text("Some content", encoding="utf-8")

    async def fake_gen(content):
        assert content == "Some content"
        return "Generated"

    manager._generate_document_summary = fake_gen
    summary = asyncio.run(manager.get_document_summary("uuid-123"))
    assert summary == "Generated"
    assert manager.metadata["uuid-123"]["summary"] == "Generated"


def test_tool_get_document_summary(tmp_path):
    bot_mod = load_bot_module()
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    manager.metadata = {"uuid-123": {"original_name": "Doc", "summary": "Hello"}}
    dummy_bot = types.SimpleNamespace(document_manager=manager)
    result = asyncio.run(bot_mod.DiscordBot._tool_get_document_summary(dummy_bot, "Doc"))
    assert result == {"summary": "Hello"}
