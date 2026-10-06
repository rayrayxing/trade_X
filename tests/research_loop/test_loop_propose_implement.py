"""Stages 1 and 2: where ideas come from, and how a draft becomes a stored, validated spec."""
import yaml

from tradex.research.loop.config import LoopConfig
from tradex.research.loop.sources import ArxivAtomSource, safe_name
from tradex.strategy.spec import StrategySpec, load_dir

from loopkit import CATALOG_ROWS, draft_file, idea_file, make_rig, spec_dict, write_spec
from test_loop_sources import ATOM


def test_buildable_catalog_entries_become_ideas_unless_something_already_covers_them(tmp_path):
    rig = make_rig(tmp_path, catalog_rows=CATALOG_ROWS, handled={"delta-handled": "has a spec (d.yaml)"})
    r = rig.run(stages=("propose",))
    ids = {i["idea_id"] for i in rig.state.ideas()}
    assert ids == {"catalog:alpha-momentum", "catalog:beta-reversal"}           # Gamma is Needs data, Delta is handled
    i = rig.state.idea("catalog:alpha-momentum")
    assert i["payload"]["arxiv"] == ["2001.00001"] and i["payload"]["assets"] == ["stocks"] and i["status"] == "new"
    assert sorted(k for k in r["propose"].items if k.startswith("catalog:")) == ["catalog:alpha-momentum", "catalog:beta-reversal"]
    rig.run(stages=("propose",), run_id="2026-W42")
    assert len(rig.state.ideas()) == 2                                          # nothing is proposed twice


def test_a_spec_that_names_the_catalog_entry_covers_it(tmp_path):
    rig = make_rig(tmp_path, catalog_rows=CATALOG_ROWS)
    rig.add_spec(spec_dict("stk-alpha", provenance={"source": "catalog: Alpha momentum (Paper A)", "author": "x"}))
    rig.add_spec(spec_dict("stk-beta", provenance={"source": "x", "catalog_id": "beta-reversal"}))
    rig.add_spec(spec_dict("stk-delta", provenance={"source": "x", "catalog_id": "delta-handled"}))
    rig.run(stages=("propose",))
    assert rig.state.ideas() == []


def test_new_ideas_per_run_are_capped_across_sources(tmp_path):
    rig = make_rig(tmp_path, catalog_rows=CATALOG_ROWS, cfg=LoopConfig(max_new_ideas=2))
    for n in range(3):
        idea_file(rig, f"f{n}")
    rig.run(stages=("propose",))
    ids = sorted(i["idea_id"] for i in rig.state.ideas())
    assert ids == ["file:f0", "file:f1"]                                        # agent files first, then the catalog, up to the cap
    rig.run(stages=("propose",), run_id="2026-W42")
    assert len(rig.state.ideas()) == 4                                          # the rest arrives next run


def test_arxiv_ideas_come_through_the_injected_getter_and_catalog_papers_are_not_repeated(tmp_path):
    src = ArxivAtomSource("cat:q-fin.TR", lambda url: ATOM)
    rows = [dict(CATALOG_ROWS[0], arxiv=["1904.04912"])]
    rig = make_rig(tmp_path, catalog_rows=rows, handled={"alpha-momentum": "x"}, extra_sources=[src])
    r = rig.run(stages=("propose",))
    assert rig.state.idea("arxiv:2601.01234")["status"] == "new"
    assert rig.state.idea("arxiv:1904.04912")["status"] == "duplicate"           # the catalog already carries that paper
    assert r["propose"].items["source:arxiv"]["duplicate"] == 1


def test_a_failing_idea_source_does_not_stop_the_others(tmp_path):
    class Down:
        name = "down"

        def fetch(self):
            raise OSError("network is down")
    rig = make_rig(tmp_path, extra_sources=[Down()])
    idea_file(rig, "ok")
    r = rig.run(stages=("propose",))
    assert r["propose"].failed == 1 and rig.state.idea("file:ok") is not None
    assert any("source:down" in a["title"] for a in rig.alerts.items)


def test_rejected_idea_files_raise_alerts(tmp_path):
    rig = make_rig(tmp_path)
    (rig.ideas_dir).mkdir(parents=True)
    (rig.ideas_dir / "bad.yaml").write_text("summary: no title\n")
    rig.run(stages=("propose",))
    assert any("idea file rejected" in a["title"] and "bad.yaml" in a["detail"] for a in rig.alerts.items)


def test_spec_files_on_disk_are_adopted_as_proposed(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.add_spec(spec_dict("stk-seed-one"), where="seeds")
    rig.run(stages=("propose",))
    assert {(r["strategy_id"], r["status"]) for r in rig.state.strategies()} == {(spec_dict()["id"], "proposed"), ("stk-seed-one", "proposed")}
    assert not any(r["unverified"] for r in rig.state.strategies())


def test_a_spec_claiming_paper_or_validated_is_flagged_and_can_never_be_promoted(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(apply=True))
    rig.add_spec(spec_dict("stk-sneaky", status="paper"))
    rig.add_spec(spec_dict("stk-sneaky2", status="validated"))
    rig.add_spec(spec_dict("stk-sneaky3", status="live"))
    r = rig.run()
    for sid in ("stk-sneaky", "stk-sneaky2", "stk-sneaky3"):
        row = rig.state.strategy(sid)
        assert row["unverified"] and "no loop evidence" in row["unverified"]
    assert rig.state.eligibility("stk-sneaky2", 1, rig.specs.content_hash("stk-sneaky2"))
    assert rig.status("stk-sneaky3") == "proposed"                              # live is not a loop status
    assert not rig.ev.calls                                                     # nothing unverified is screened or walked forward
    assert r["promote"].items == {} or all(v.get("outcome") != "promoted" for v in r["promote"].items.values())
    crit = [a for a in rig.alerts.items if a["level"] == "critical" and "claims" in a["title"]]
    assert len(crit) == 3
    assert rig.status("stk-sneaky") == "paper"                                  # the loop records what it found, flagged
    assert any("outside the loop" in x for x in rig.state.eligibility("stk-sneaky", 1, "h"))


def test_a_spec_file_edited_to_say_paper_is_reported(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    rig.run(stages=("propose", "screen"))
    rig.specs.set_status("stk-rsi-test", "paper")                               # someone edits the file by hand
    rig.run(stages=("propose",), run_id="2026-W42")
    assert any(a["level"] == "critical" and "the loop has screened" in a["title"] for a in rig.alerts.items)
    assert rig.status("stk-rsi-test") == "screened"                             # the database is the record


# --- implement --------------------------------------------------------------------------------------

def test_a_valid_draft_is_written_as_proposed_with_provenance(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "gap", arxiv=["2601.00001"])
    draft_file(rig, "file:gap", spec_dict("stk-gap-fade", status="paper", stats={"hit_rate": 0.9},
                                          provenance={"source": "agent draft", "author": "strategy_designer"}))
    r = rig.run(stages=("propose", "implement"))
    out = r["implement"].items["idea:file:gap"]
    assert out["outcome"] == "specced" and out["strategy_id"] == "stk-gap-fade"
    raw = yaml.safe_load((rig.root / "strategies" / "proposed" / "stk-gap-fade.yaml").read_text())
    assert raw["status"] == "proposed" and "stats" not in raw                    # claims stripped
    assert raw["provenance"]["idea_id"] == "file:gap" and raw["provenance"]["arxiv"] == ["2601.00001"]
    assert raw["provenance"]["loop_run"] == "2026-W41" and raw["provenance"]["author"] == "strategy_designer"
    assert any("forced to proposed" in n for n in out["notes"]) and any("stats" in n for n in out["notes"])
    assert rig.status("stk-gap-fade") == "proposed" and rig.state.idea("file:gap")["status"] == "specced"
    assert rig.state.strategy("stk-gap-fade")["idea_id"] == "file:gap"


def test_written_specs_pass_the_check_that_ci_runs(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "one")
    draft_file(rig, "file:one", spec_dict("stk-one-new"))
    rig.run(stages=("propose", "implement"))
    specs = load_dir(rig.root / "strategies")
    assert [s.id for s in specs] == ["stk-one-new"] and specs[0].validate() == []


def test_an_invalid_draft_is_recorded_and_nothing_is_written(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "bad")
    draft_file(rig, "file:bad", spec_dict("stk-bad", entry={"long": "close > nonexistent_feature"}, family="astrology"))
    r = rig.run(stages=("propose", "implement"))
    out = r["implement"].items["idea:file:bad"]
    assert out["outcome"] == "spec_invalid" and any("undefined names" in e for e in out["errors"])
    assert any("family" in e for e in out["errors"])
    assert not list((rig.root / "strategies" / "proposed").glob("*.yaml"))
    assert rig.state.idea("file:bad")["status"] == "spec_invalid"


def test_a_fixed_draft_is_picked_up_but_an_unchanged_invalid_one_is_not_retried(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "x")
    draft_file(rig, "file:x", spec_dict("stk-x-draft", entry={"long": "nope > 1"}))
    rig.run(stages=("propose", "implement"))
    again = rig.run(stages=("implement",), run_id="2026-W42")
    assert again["implement"].items["idea:file:x"].get("unchanged") is True
    draft_file(rig, "file:x", spec_dict("stk-x-draft"))
    fixed = rig.run(stages=("implement",), run_id="2026-W43")
    assert fixed["implement"].items["idea:file:x"]["outcome"] == "specced"


def test_an_unreadable_draft_file_is_an_invalid_draft(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "y")
    rig.drafts_dir.mkdir(parents=True)
    (rig.drafts_dir / f"{safe_name('file:y')}.yaml").write_text("- a list, not a spec\n")
    r = rig.run(stages=("propose", "implement"))
    assert r["implement"].items["idea:file:y"]["outcome"] == "spec_invalid"


def test_ideas_without_a_draft_wait_and_are_reported_not_invented(tmp_path):
    rig = make_rig(tmp_path, catalog_rows=CATALOG_ROWS, handled={"delta-handled": "has a spec (d.yaml)"})
    r = rig.run(stages=("propose", "implement", "report"))
    assert {v["outcome"] for v in r["implement"].items.values()} == {"awaiting_spec"}
    assert rig.state.strategies() == [] and not list((rig.root / "strategies" / "proposed").glob("*.yaml"))
    assert len(rig.state.ideas("awaiting_spec")) == 2
    md = (rig.out_dir / "weekly-2026-W41.md").read_text()
    assert "Ideas waiting for a draft spec" in md and "catalog:alpha-momentum" in md


def test_a_draft_cannot_take_over_an_existing_strategy_id(tmp_path):
    rig = make_rig(tmp_path)
    rig.add_spec()
    idea_file(rig, "dup")
    draft_file(rig, "file:dup", spec_dict())                                    # same id as the spec already on disk
    r = rig.run(stages=("propose", "implement"))
    assert "already exists" in " ".join(r["implement"].items["idea:file:dup"]["errors"])


def test_drafts_are_limited_per_run_but_waiting_ideas_do_not_use_the_budget(tmp_path):
    rig = make_rig(tmp_path, cfg=LoopConfig(max_drafts_per_run=1))
    for n in ("a", "b", "c"):
        idea_file(rig, n)
    draft_file(rig, "file:b", spec_dict("stk-b-new"))
    draft_file(rig, "file:c", spec_dict("stk-c-new"))
    rig.run(stages=("propose", "implement"))
    assert {r["strategy_id"] for r in rig.state.strategies()} == {"stk-b-new"}   # a waits, b is written, c is left for next run
    rig.run(stages=("implement",), run_id="2026-W42")
    assert {r["strategy_id"] for r in rig.state.strategies()} == {"stk-b-new", "stk-c-new"}


def test_a_drafted_spec_is_then_screened_like_any_other(tmp_path):
    rig = make_rig(tmp_path)
    idea_file(rig, "z")
    draft_file(rig, "file:z", spec_dict("stk-z-new"))
    rig.run()
    assert rig.status("stk-z-new") == "validated" and rig.file_status("stk-z-new") == "validated"
    assert StrategySpec.load(rig.specs.get("stk-z-new").path).status == "validated"
