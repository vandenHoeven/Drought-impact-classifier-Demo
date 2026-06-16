from __future__ import annotations

import re
import unicodedata
from typing import Any

PLACE_TYPE_RANK: dict[str, int] = {
    "statistical": 100,
    "region": 95,
    "safety_region": 95,
    "state": 90,
    "province": 90,
    "county": 85,
    "state_district": 85,
    "protected_area": 80,
    "country": 70,
    "city": 60,
    "town": 55,
    "municipality": 55,
    "village": 50,
    "canal": 41,
    "river": 41,
    "locality": 35,
    "hamlet": 10,
    "tourism": 9,
    "road": 5,
    "railway": 5,
    "building": 1,
    "amenity": 1,
    "stream": 1,
    "waterway": 1,
    "shop": 1,
    "construction": 1,
    "office": 1,
}
DEFAULT_PLACE_TYPE_RANK = 50
LOW_QUALITY_TYPES = frozenset({
    "hamlet",
    "tourism",
    "road",
    "railway",
    "building",
    "amenity",
    "stream",
    "waterway",
    "shop",
    "construction",
    "office",
})

FORCED_LOW_QUALITY_LOCATIONS = frozenset({"de jordaan"})


def _override_location_key(location: str) -> str:
    text = (location or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def place_type_rank(addresstype: str) -> int:
    return PLACE_TYPE_RANK.get((addresstype or "").lower(), DEFAULT_PLACE_TYPE_RANK)


def is_low_quality_type(addresstype: str) -> bool:
    return (addresstype or "").lower() in LOW_QUALITY_TYPES


def apply_location_geocode_override(
    location: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    norm = _override_location_key(location)
    if norm in FORCED_LOW_QUALITY_LOCATIONS:
        return {
            **fields,
            "geocoded_place_rank": 1,
            "geocoded_is_low_quality": True,
        }
    return fields


def enrich_geocode_fields(
    place_type: str | None,
    location: str | None = None,
) -> dict[str, int | bool | None]:
    if not place_type or not str(place_type).strip():
        fields: dict[str, int | bool | None] = {
            "geocoded_place_rank": None,
            "geocoded_is_low_quality": False,
        }
    else:
        normalized = str(place_type).strip()
        fields = {
            "geocoded_place_rank": place_type_rank(normalized),
            "geocoded_is_low_quality": is_low_quality_type(normalized),
        }
    if location:
        fields = apply_location_geocode_override(location, fields)
    return fields


def format_rank_table_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "Rank": rank,
            "Place type": addresstype,
            "Low quality": "Yes" if addresstype in LOW_QUALITY_TYPES else "No",
        }
        for addresstype, rank in PLACE_TYPE_RANK.items()
    ]
    rows.sort(key=lambda row: (-row["Rank"], row["Place type"]))
    rows.append({
        "Rank": DEFAULT_PLACE_TYPE_RANK,
        "Place type": "(unknown / other types)",
        "Low quality": "No",
    })
    return rows


def format_location_override_notes() -> list[str]:
    return [
        f'"{name}" → forced rank 1, low-quality'
        for name in sorted(FORCED_LOW_QUALITY_LOCATIONS)
    ]
