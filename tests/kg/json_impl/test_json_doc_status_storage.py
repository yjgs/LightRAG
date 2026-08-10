import pytest

from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(status: str, file_path: str) -> dict:
    return {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.mark.asyncio
async def test_get_docs_paginated_with_status_filters(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("preprocessed", "a.pdf"),
                "doc-2": _doc("parsing", "b.pdf"),
                "doc-3": _doc("analyzing", "c.pdf"),
                "doc-4": _doc("processed", "d.pdf"),
            }
        )

    docs, total = await storage.get_docs_paginated(
        status_filter=DocStatus.PROCESSED,
        status_filters=[
            DocStatus.PREPROCESSED,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
        ],
        page=1,
        page_size=10,
        sort_field="id",
        sort_direction="asc",
    )

    assert total == 3
    assert [doc_id for doc_id, _ in docs] == ["doc-1", "doc-2", "doc-3"]
    assert [doc.status for _, doc in docs] == [
        DocStatus.PREPROCESSED,
        DocStatus.PARSING,
        DocStatus.ANALYZING,
    ]


@pytest.mark.asyncio
async def test_doc_status_upsert_preserves_caller_file_path(tmp_path):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc-1": _doc(
                DocStatus.PENDING.value,
                "/tmp/uploads/report.[native-Fi].pdf",
            )
        }
    )

    assert (await storage.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_doc_status_basename_lookup_requires_canonical_stored_path(tmp_path):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data["doc-1"] = _doc(
            DocStatus.PROCESSED.value,
            "report.[native].pdf",
        )

    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_json_kv_upsert_preserves_caller_file_paths(tmp_path):
    full_docs = JsonKVStorage(
        namespace=NameSpace.KV_STORE_FULL_DOCS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    text_chunks = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await full_docs.initialize()
    await text_chunks.initialize()

    await full_docs.upsert(
        {
            "doc-1": {
                "content": "full text",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )
    await text_chunks.upsert(
        {
            "chunk-1": {
                "content": "chunk text",
                "tokens": 2,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )

    assert (await full_docs.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert (await text_chunks.get_by_id("chunk-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )


@pytest.mark.asyncio
async def test_get_docs_by_statuses_strict_raises_on_bad_record(tmp_path):
    """Strict scheduling contract: a record that cannot be converted raises;
    the relaxed default keeps the historical skip-and-log behavior."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-good": _doc("failed", "good.pdf"),
                # Missing required fields → DocProcessingStatus(**data) raises.
                "doc-bad": {"status": "failed"},
            }
        )

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"doc-good"}

    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


@pytest.mark.asyncio
async def test_get_docs_by_statuses_missing_status_skips_relaxed_raises_strict(
    tmp_path,
):
    """A record missing its ``status`` key is just another undeserializable
    record: skipped in relaxed mode, raised under strict.  Regression guard for
    reading ``v["status"]`` OUTSIDE the try, which crashed every relaxed caller
    on such a record instead of skipping it."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-good": _doc("failed", "good.pdf"),
                "doc-nostatus": {"file_path": "x.pdf"},  # no "status" key at all
            }
        )

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"doc-good"}  # skipped, not crashed

    with pytest.raises(KeyError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


@pytest.mark.asyncio
async def test_get_docs_paginated_search_by_file_path_case_insensitive(tmp_path):
    """search matches file_path case-insensitively."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("processed", "report_2026.pdf"),
                "doc-2": _doc("processed", "SUMMARY.txt"),
                "doc-3": _doc("processed", "notes.md"),
            }
        )

    docs, total = await storage.get_docs_paginated(
        search="REPORT", page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total == 1
    assert [doc_id for doc_id, _ in docs] == ["doc-1"]

    docs, total = await storage.get_docs_paginated(
        search="summary", page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total == 1
    assert [doc_id for doc_id, _ in docs] == ["doc-2"]


@pytest.mark.asyncio
async def test_get_docs_paginated_search_by_doc_id(tmp_path):
    """search matches the document id as well."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("processed", "a.pdf"),
                "doc-2": _doc("processed", "b.pdf"),
                "doc-3": _doc("processed", "c.pdf"),
            }
        )

    docs, total = await storage.get_docs_paginated(
        search="DOC-2", page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total == 1
    assert [doc_id for doc_id, _ in docs] == ["doc-2"]


@pytest.mark.asyncio
async def test_get_docs_paginated_search_empty_or_none_returns_all(tmp_path):
    """search=None or "" disables filtering."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("processed", "a.pdf"),
                "doc-2": _doc("processed", "b.pdf"),
            }
        )

    docs_none, total_none = await storage.get_docs_paginated(
        search=None, page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total_none == 2

    docs_empty, total_empty = await storage.get_docs_paginated(
        search="", page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total_empty == 2
    assert len(docs_empty) == 2


@pytest.mark.asyncio
async def test_get_docs_paginated_search_no_match(tmp_path):
    """No match returns an empty page with total 0."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("processed", "a.pdf"),
                "doc-2": _doc("processed", "b.pdf"),
            }
        )

    docs, total = await storage.get_docs_paginated(
        search="zzz-not-exist", page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert total == 0
    assert docs == []


@pytest.mark.asyncio
async def test_get_docs_paginated_search_with_status_and_pagination(tmp_path):
    """search composes with status filter and pagination: total reflects the
    filtered set and pages slice it correctly."""
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                **{
                    f"doc-p{i:02d}": _doc("processed", f"report-{i:02d}.pdf")
                    for i in range(1, 13)  # doc-p01..doc-p12 (12 processed)
                },
                **{
                    f"doc-f{i:02d}": _doc("failed", f"report-{i:02d}.pdf")
                    for i in range(1, 13)  # doc-f01..doc-f12 (12 failed)
                },
            }
        )

    # search "report" + status processed → 12 docs (doc-p01..doc-p12).
    # page_size is clamped to >= 10 by the backend, so use page_size=10.
    # Zero-padded ids keep the default string (lexicographic) sort numeric.
    docs_page1, total = await storage.get_docs_paginated(
        search="report",
        status_filter=DocStatus.PROCESSED,
        page=1,
        page_size=10,
        sort_field="id",
        sort_direction="asc",
    )
    assert total == 12
    assert len(docs_page1) == 10
    assert [doc_id for doc_id, _ in docs_page1] == [
        f"doc-p{i:02d}" for i in range(1, 11)
    ]

    docs_page2, _ = await storage.get_docs_paginated(
        search="report",
        status_filter=DocStatus.PROCESSED,
        page=2,
        page_size=10,
        sort_field="id",
        sort_direction="asc",
    )
    assert [doc_id for doc_id, _ in docs_page2] == ["doc-p11", "doc-p12"]
