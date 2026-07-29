"""Parser adapters that normalize native output into the Spatial Document IR."""

from src.complex_document.parsers.base import (
    DocumentParserAdapter,
    ParseRequest,
    ParserUnavailable,
)
from src.complex_document.parsers.hybrid_table_router_parser import (
    HybridTableRouterAdapter,
)
from src.complex_document.parsers.liteparse_parser import LiteParseAdapter
from src.complex_document.parsers.liteparse_table_parser import LiteParseTableAdapter
from src.complex_document.parsers.paddle_layout_parser import PaddleLayoutAdapter
from src.complex_document.parsers.pymupdf_parser import PyMuPDFAdapter
from src.complex_document.parsers.targeted_vlm_router_parser import (
    TargetedVLMRouterAdapter,
)
from src.complex_document.parsers.vlm_parser import QwenVLMParserAdapter

__all__ = [
    "DocumentParserAdapter",
    "HybridTableRouterAdapter",
    "LiteParseAdapter",
    "LiteParseTableAdapter",
    "PaddleLayoutAdapter",
    "ParseRequest",
    "ParserUnavailable",
    "PyMuPDFAdapter",
    "QwenVLMParserAdapter",
    "TargetedVLMRouterAdapter",
]
