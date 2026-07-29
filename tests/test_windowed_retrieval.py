from src.complex_document.chunking import Chunk
from src.complex_document.retrieval import WindowedCharNgramRetriever


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        markdown=text,
        pages=[1],
        bboxes=[],
        section_path=["統計表"],
        parser_name="test",
        parser_version="1",
        element_ids=[],
    )


def test_windowed_retriever_finds_local_row_inside_long_atomic_table():
    target = _chunk(
        "target",
        ("無關欄位 " * 200)
        + "113年 桃園機場 入出國境人數 3,416,816"
        + (" 其他資料" * 200),
    )
    distractor = _chunk("distractor", "桃園機場年度簡介與交通方式")
    retriever = WindowedCharNgramRetriever(
        window_size=120, overlap=40, global_weight=0
    )
    hits = retriever.retrieve(
        "113年桃園機場入出國境人數", [distractor, target], k=2
    )
    assert hits[0].chunk.chunk_id == "target"


def test_windowed_retriever_validates_configuration():
    try:
        WindowedCharNgramRetriever(window_size=80, overlap=80)
    except ValueError as exc:
        assert "window_size" in str(exc)
    else:
        raise AssertionError("invalid overlap should fail")
