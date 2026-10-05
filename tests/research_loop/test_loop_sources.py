from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tradex.research import catalog as catalog_mod
from tradex.research.loop.ports import Idea
from tradex.research.loop.sources import (ArxivAtomSource, DraftError, FileCatalog, FileIdeaSource, FileSpecDrafter,
                                          clean, parse_atom, safe_name)

ATOM = (Path(__file__).resolve().parents[1] / "fixtures" / "research_loop" / "arxiv_atom.xml").read_text()


def test_atom_results_become_ideas():
    ideas = parse_atom(ATOM)
    assert [i.id for i in ideas] == ["arxiv:2601.01234", "arxiv:1904.04912"]          # the old-style id is skipped
    a = ideas[0]
    assert a.title == "Overnight Drift in Sector ETFs"                                  # whitespace collapsed
    assert a.url == "https://arxiv.org/abs/2601.01234" and a.arxiv == ("2601.01234",)
    assert a.published == "2026-01-04T09:00:00Z" and a.extra["categories"] == ["q-fin.TR", "q-fin.PM"]
    assert a.source == "arxiv"


def test_text_from_a_paper_is_data_it_cannot_set_a_status():
    a = parse_atom(ATOM)[0]
    assert "Ignore previous instructions" in a.summary                                  # kept as text, nothing reads it as a command
    assert not hasattr(a, "status")


@pytest.mark.parametrize("payload", ['<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><feed/>',
                                     '<!doctype feed><feed xmlns="http://www.w3.org/2005/Atom"/>'])
def test_a_response_with_a_dtd_is_refused(payload):
    with pytest.raises(ValueError, match="DTD"):
        parse_atom(payload)


def test_the_source_builds_an_https_query_and_calls_only_the_injected_getter():
    calls = []
    src = ArxivAtomSource("cat:q-fin.TR AND all:momentum", lambda url: calls.append(url) or ATOM, max_results=7)
    ideas = src.fetch()
    assert len(ideas) == 2 and len(calls) == 1
    assert calls[0].startswith("https://export.arxiv.org/api/query?") and "max_results=7" in calls[0]
    assert "search_query=cat%3Aq-fin.TR+AND+all%3Amomentum" in calls[0]


def test_clean_strips_control_characters_and_caps_length():
    assert clean("a\x00b\x1bc\td\n", 50) == "abc d"
    assert len(clean("x" * 5000, 100)) == 100
    assert clean(None, 10) == ""


def test_safe_name_keeps_ids_out_of_paths():
    assert safe_name("arxiv:2601.01234") == "arxiv__2601.01234"
    assert "/" not in safe_name("file:../../etc/passwd")


def write(d, name, doc):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc))


def test_idea_files_are_read_and_bad_ones_are_reported_not_raised(tmp_path):
    d = tmp_path / "ideas"
    write(d, "good.yaml", {"title": "Weekend gap fade", "summary": "s", "arxiv": ["2601.00001"], "assets": ["Stocks", "etfs"],
                           "family": "mean_reversion"})
    write(d, "no-title.yaml", {"summary": "x"})
    write(d, "bad-arxiv.yaml", {"title": "t", "arxiv": ["not-an-id"]})
    write(d, "bad-asset.yaml", {"title": "t", "assets": ["crypto"]})
    write(d, "broken.yaml", "title: [unclosed")
    write(d, "list.yaml", "- a\n- b\n")
    write(d, "notes.txt", "ignored")
    (d / "specs").mkdir()
    src = FileIdeaSource(d)
    ideas = src.fetch()
    assert [i.id for i in ideas] == ["file:good"]
    g = ideas[0]
    assert g.source == "agent-file" and g.assets == ("stocks", "etfs") and g.arxiv == ("2601.00001",)
    assert len(src.errors) == 5 and any("bad-arxiv.yaml" in e for e in src.errors)


def test_a_missing_ideas_folder_is_empty(tmp_path):
    assert FileIdeaSource(tmp_path / "none").fetch() == []


def test_idea_file_text_is_capped(tmp_path):
    d = tmp_path / "i"
    write(d, "long.yaml", {"title": "T" * 999, "summary": "S" * 99999})
    i = FileIdeaSource(d).fetch()[0]
    assert len(i.title) == 200 and len(i.summary) == 4000


def test_spec_drafter_finds_drafts_by_idea_or_catalog_id(tmp_path):
    d = tmp_path / "specs"
    write(d, "arxiv__2601.01234.yaml", {"id": "stk-x"})
    write(d, "alpha-momentum.yaml", {"id": "stk-y"})
    dr = FileSpecDrafter(d)
    assert dr.draft(Idea(id="arxiv:2601.01234", source="arxiv", title="t"))["id"] == "stk-x"
    assert dr.draft(Idea(id="catalog:alpha-momentum", source="catalog", title="t", catalog_id="alpha-momentum"))["id"] == "stk-y"
    assert dr.draft(Idea(id="arxiv:0000.00000", source="arxiv", title="t")) is None
    write(d, "file__bad.yaml", "- not a mapping")
    with pytest.raises(DraftError):
        dr.draft(Idea(id="file:bad", source="agent-file", title="t"))


def test_catalog_handled_covers_planned_entries_and_gate_verdicts():
    plans = [SimpleNamespace(catalog_id="a", spec=Path("strategies/seeds/x.yaml"), verdict=None, why=""),
             SimpleNamespace(catalog_id="b", spec=None, verdict="not built", why="needs a trained model")]
    h = FileCatalog(plans=plans).handled()
    assert h["a"] == "has a spec (x.yaml)" and h["b"] == "not built: needs a trained model"


def test_the_real_catalog_leaves_nothing_unhandled_among_its_build_entries():
    """Every Build entry has a spec or a recorded reason it cannot be one, so the catalog adds no new ideas today."""
    fc = FileCatalog()
    entries = fc.entries()
    handled = fc.handled()
    assert {e.id for e in entries if e.status_kind == "Build"} <= set(handled)
    assert isinstance(entries[0], catalog_mod.CatalogEntry)
