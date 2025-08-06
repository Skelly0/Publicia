import importlib.util
from pathlib import Path
import sys
import types
import json


def load_document_module():
    module_path = Path(__file__).resolve().parents[1] / "managers" / "documents.py"
    # Stub google.generativeai to avoid heavy import and API calls
    if 'google.generativeai' not in sys.modules:
        google_pkg = types.ModuleType('google')
        genai_mod = types.ModuleType('google.generativeai')
        def configure(api_key=None):
            pass
        genai_mod.configure = configure
        google_pkg.generativeai = genai_mod
        sys.modules['google'] = google_pkg
        sys.modules['google.generativeai'] = genai_mod
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
        utils_pkg.logging = logging_pkg
        sys.modules['utils'] = utils_pkg
        sys.modules['utils.logging'] = logging_pkg
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


def test_resolve_doc_identifier(tmp_path):
    manager = DocumentManager(base_dir=str(tmp_path), config=DummyConfig())
    doc_uuid = "abc-123"
    manager.metadata = {doc_uuid: {"original_name": "TestDoc"}}

    tracked = tmp_path / "tracked_google_docs.json"
    tracked.write_text(json.dumps([{"google_doc_id": "GID123", "internal_doc_uuid": doc_uuid}]))

    assert manager.resolve_doc_identifier(doc_uuid) == doc_uuid
    assert manager.resolve_doc_identifier("TestDoc") == doc_uuid
    assert manager.resolve_doc_identifier("GID123") == doc_uuid
