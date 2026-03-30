"""Tests for src.utils.io."""

import pytest

from src.utils.io import (
    INTERVAL_PERIOD_CAP,
    cap_period_for_interval,
    fetch_universe_ohlcv,
    parse_period_days,
)


def test_parse_period_days():
    assert parse_period_days("7d") == 7
    assert parse_period_days("30d") == 30
    assert parse_period_days("1y") == 252
    assert parse_period_days("2y") == 504
    assert parse_period_days("1mo") == 21
    assert parse_period_days("1wk") == 5
    assert parse_period_days("unknown") == 252


def test_cap_period_for_interval():
    # No cap for 1d
    assert cap_period_for_interval("1d", "5y") == "5y"
    # 1m capped to 7d
    assert cap_period_for_interval("1m", "30d") == "7d"
    assert cap_period_for_interval("1m", "5d") == "5d"
    # Override
    assert cap_period_for_interval("1m", "30d", period_override=True) == "30d"


def test_interval_period_cap_keys():
    assert "1m" in INTERVAL_PERIOD_CAP
    assert "5m" in INTERVAL_PERIOD_CAP
    assert INTERVAL_PERIOD_CAP["1m"] == "7d"
