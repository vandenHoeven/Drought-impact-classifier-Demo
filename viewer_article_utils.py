from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

_SOURCE_FILE_EXT = re.compile(r"\.(docx?|DOCX?)$")


def decode_article_title(text: str) -> str:
    if not text:
        return ""
    return html.unescape(str(text).strip())


def _article_id_from_row(row: Any) -> str:
    for key in ("id", "article_id"):
        value = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _title_from_source_file(row: Any) -> str:
    source_file = row.get("source_file") if hasattr(row, "get") else getattr(row, "source_file", None)
    if source_file is None or not str(source_file).strip():
        return ""
    cleaned = _SOURCE_FILE_EXT.sub("", str(source_file).strip())
    return decode_article_title(cleaned)


def resolve_article_title(row: Any, article_lookup: dict[str, dict] | None) -> str:
    article_id = _article_id_from_row(row)
    if article_lookup and article_id:
        article = article_lookup.get(article_id, {})
        title = decode_article_title(article.get("title", ""))
        if title:
            return title

    source_title = _title_from_source_file(row)
    if source_title:
        return source_title

    return article_id


def attach_article_titles(
    df: pd.DataFrame,
    article_lookup: dict[str, dict] | None,
) -> pd.DataFrame:
    display_df = df.copy()
    display_df["article_title"] = [
        resolve_article_title(row, article_lookup) for _, row in display_df.iterrows()
    ]
    return display_df
