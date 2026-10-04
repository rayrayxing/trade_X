import pytest

from tradex.research import catalog


def test_repo_catalog_loads():
    entries = catalog.load()
    assert len(entries) == 77
    assert len({e.id for e in entries}) == 77
    kinds = {e.status_kind for e in entries}
    assert kinds <= set(catalog.STATUS_KINDS)
    builds = catalog.by_status(entries, "Build")
    assert len(builds) >= 20 and all(e.is_build for e in builds)
    tick = next(e for e in entries if e.name.startswith("Large-tick"))
    assert tick.status_kind == "Build" and tick.status_note == "filter for all trend strategies"
    assert tick.assets == ("forex", "stocks")
    assert any(e.status_kind == "Seed" for e in entries)


def _row(**kw):
    r = {"family": "f", "name": "A thing", "asset": "FX, stocks", "horizon": "Days", "data": "Daily bars",
         "source": "x", "data_today": "Yes", "status": "Build", "arxiv": ["2012.07149"]}
    return r | kw


def test_parse_rejects_bad_entries():
    assert catalog.parse([_row()])[0].id == "a-thing"
    with pytest.raises(catalog.CatalogError, match="status"):
        catalog.parse([_row(status="Maybe later")])
    with pytest.raises(catalog.CatalogError, match="asset"):
        catalog.parse([_row(asset="Crypto")])
    with pytest.raises(catalog.CatalogError, match="missing"):
        catalog.parse([_row(source="")])
    with pytest.raises(catalog.CatalogError, match="duplicate"):
        catalog.parse([_row(), _row()])
    with pytest.raises(catalog.CatalogError, match="arxiv"):
        catalog.parse([_row(arxiv=["not-an-id"])])
