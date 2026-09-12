"""Sampler V3.8 RefMod (ComfyUI-MiniMaxH3Mod) bridge contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ComfyUI_H3_Continuum_Join import refmod_bridge
from ComfyUI_H3_Continuum_Join.refmod_bridge import (
    BRIDGE_WRAPPER_KEY,
    CURVE_DIRECTIONS,
    CURVE_SHAPES,
    RefModBridgeError,
    SCRAMBLE_MODES,
    attach_refmod_blocks,
    build_refmod_blocks,
    bundle_items,
    format_refmod_status,
    normalize_settings,
    resolve_refmod_pack,
)
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)


ROOT = Path(__file__).resolve().parents[1]
REFMOD_WIDGETS = (
    "refmod_retention",
    "refmod_curve_direction",
    "refmod_curve_shape",
    "refmod_curve_value",
    "refmod_scramble_seed",
    "refmod_scramble_mode",
    "refmod_scramble_keep",
    "refmod_max_tokens",
    "refmod_override",
)


class _Mod:
    """Duck-typed stand-in for the pack's H3RefMod dataclass."""

    def __init__(self, name, *, latent_t=1, kind="image", config=None, tokens=16):
        self.name = name
        self.kind = kind
        self.latent_t = latent_t
        self.latent_h = 8
        self.latent_w = 8
        self.config = config or {}
        self.token_count = tokens
        self.calls = []

    def ref_block(self, strength=1.0, curve=None):
        self.calls.append((strength, curve))
        if strength <= 0.0:
            return None
        return {
            "kind": self.kind,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "latent": f"latent:{self.name}@{strength:.2f}",
            **({"latent_t": self.latent_t, "ref_audio_t": 0, "audio_latent": None}
               if self.kind == "video" else {}),
        }


class _Model:
    """Minimal ModelPatcher double exposing the public wrapper API."""

    def __init__(self, wrappers=None):
        self.wrappers = {k: dict(v) for k, v in (wrappers or {}).items()}

    def clone(self):
        return _Model(self.wrappers)

    def remove_wrappers_with_key(self, group, key):
        self.wrappers.get(group, {}).pop(key, None)

    def add_wrapper_with_key(self, group, key, fn):
        self.wrappers.setdefault(group, {}).setdefault(key, []).append(fn)


class _Executor:
    def __init__(self, guider):
        self.class_obj = guider
        self.seen = None

    def __call__(self, *args, **kwargs):
        self.seen = {k: [dict(e) for e in v] for k, v in self.class_obj.conds.items()}
        return "sampled"


# ---------------------------------------------------------------- settings


def test_normalize_settings_clamps_and_falls_back_without_stopping():
    settings = normalize_settings(
        retention=1.7,
        curve_direction="bogus",
        curve_shape="nope",
        curve_value=-3,
        scramble_seed="x",
        scramble_mode="???",
        scramble_keep=0,
        max_total_tokens=-5,
        override="yes",
    )
    assert settings["retention"] == 1.0
    assert settings["curve_direction"] == "constant"
    assert settings["curve_shape"] == "linear"
    assert settings["curve_value"] == 0.0
    assert settings["scramble_seed"] == -1
    assert settings["scramble_mode"] == "shuffle"
    assert settings["scramble_keep"] == 1
    assert settings["max_total_tokens"] == 0
    assert settings["override"] is True
    assert len(settings["notes"]) == 3


def test_bundle_items_rejects_non_bundles_clearly():
    assert bundle_items(None) == []
    with pytest.raises(RefModBridgeError):
        bundle_items("not a bundle")
    with pytest.raises(RefModBridgeError):
        bundle_items([("no ref_block method", 1.0)])
    items = bundle_items([(_Mod("a"), 2.0), (_Mod("b"), 0.25)])
    assert [s for _, s in items] == [1.0, 0.25]


# ---------------------------------------------------------- block building


def test_fallback_blocks_apply_retention_row_strength_and_marker():
    a, b = _Mod("a"), _Mod("b")
    blocks, summary = build_refmod_blocks(
        [(a, 1.0), (b, 0.5)], normalize_settings(retention=0.7), pack=None
    )
    assert [blk["latent"] for blk in blocks] == ["latent:a@0.70", "latent:b@0.35"]
    assert all(blk["refmod"] is True for blk in blocks)
    assert summary["helper"] == "continuum fallback"
    assert summary["block_count"] == 2
    assert summary["token_count"] == 2 * (4 * 4)


def test_fallback_single_frame_mod_uses_curve_value_as_cap():
    image, video = _Mod("img", latent_t=1), _Mod("vid", latent_t=4, kind="video")
    settings = normalize_settings(curve_direction="concept_at_end", curve_shape="ease", curve_value=0.4)
    blocks, _ = build_refmod_blocks([(image, 1.0), (video, 1.0)], settings, pack=None)
    assert image.calls == [(0.4, None)]
    assert video.calls == [(1.0, ("concept_at_end", "ease", 0.4))]
    assert len(blocks) == 2


def test_fallback_scramble_is_seeded_and_subset_keeps_requested_count():
    mods = [(_Mod(str(i)), 1.0) for i in range(6)]
    first, _ = build_refmod_blocks(mods, normalize_settings(scramble_seed=7), pack=None)
    second, _ = build_refmod_blocks(mods, normalize_settings(scramble_seed=7), pack=None)
    assert [b["latent"] for b in first] == [b["latent"] for b in second]
    assert sorted(b["latent"] for b in first) == sorted(f"latent:{i}@1.00" for i in range(6))
    subset, _ = build_refmod_blocks(
        mods, normalize_settings(scramble_seed=3, scramble_mode="subset", scramble_keep=2), pack=None
    )
    assert len(subset) == 2
    off, _ = build_refmod_blocks(mods, normalize_settings(scramble_seed=-1), pack=None)
    assert [b["latent"] for b in off] == [f"latent:{i}@1.00" for i in range(6)]


def test_fallback_token_budget_matches_apply_contract():
    mods = [(_Mod("a", tokens=100), 1.0), (_Mod("b", tokens=100), 1.0)]
    build_refmod_blocks(mods, normalize_settings(max_total_tokens=0), pack=None)
    build_refmod_blocks(mods, normalize_settings(max_total_tokens=200), pack=None)
    with pytest.raises(RefModBridgeError):
        build_refmod_blocks(mods, normalize_settings(max_total_tokens=199), pack=None)


def test_override_uses_first_saved_config_and_notes_when_absent():
    plain = _Mod("plain")
    tuned = _Mod("tuned", latent_t=3, kind="video",
                 config={"retention": 0.4, "curve": ["concept_at_start", "sigmoid", 0.6]})
    blocks, summary = build_refmod_blocks(
        [(plain, 1.0), (tuned, 1.0)], normalize_settings(retention=1.0, override=True), pack=None
    )
    settings = summary["settings"]
    assert settings["retention"] == 0.4
    assert (settings["curve_direction"], settings["curve_shape"], settings["curve_value"]) == (
        "concept_at_start", "sigmoid", 0.6,
    )
    assert "tuned" in settings["curve_source"]
    assert tuned.calls == [(0.4, ("concept_at_start", "sigmoid", 0.6))]
    assert len(blocks) == 2

    _, summary = build_refmod_blocks([(plain, 1.0)], normalize_settings(override=True), pack=None)
    assert any("override is on" in note for note in summary["notes"])
    assert summary["settings"]["retention"] == 1.0


def test_invalid_saved_config_is_ignored_not_trusted():
    junk = _Mod("junk", config={"retention": "high", "curve": ["evil", "linear", 1.0]})
    _, summary = build_refmod_blocks([(junk, 1.0)], normalize_settings(override=True), pack=None)
    assert summary["settings"]["curve_source"] == "sampler widgets"
    assert summary["settings"]["retention"] == 1.0


def test_pack_helper_is_preferred_when_installed():
    calls = {}

    def _ref_blocks(mods, retention, curve=None, seed=-1, scramble_mode="legacy_subset",
                    scramble_keep=1, max_total_tokens=0):
        calls.update(dict(mods=list(mods), retention=retention, curve=curve, seed=seed,
                          scramble_mode=scramble_mode, scramble_keep=scramble_keep,
                          max_total_tokens=max_total_tokens))
        return [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "pack", "refmod": True}]

    pack = types.SimpleNamespace(__name__="fake_pack.nodes", _ref_blocks=_ref_blocks)
    mod = _Mod("m")
    blocks, summary = build_refmod_blocks(
        [(mod, 0.8)],
        normalize_settings(retention=0.5, curve_direction="concept_at_middle", curve_shape="bump",
                           curve_value=0.9, scramble_seed=11, scramble_mode="subset",
                           scramble_keep=2, max_total_tokens=64),
        pack=pack,
    )
    assert blocks == [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "pack", "refmod": True}]
    assert summary["helper"] == "fake_pack.nodes._ref_blocks"
    assert calls["mods"] == [(mod, 0.8)]
    assert calls["retention"] == 0.5
    assert calls["curve"] == ("concept_at_middle", "bump", 0.9)
    assert (calls["seed"], calls["scramble_mode"], calls["scramble_keep"], calls["max_total_tokens"]) == (
        11, "subset", 2, 64,
    )
    assert mod.calls == []


def test_resolve_refmod_pack_discovers_by_capability_only(monkeypatch):
    fake = types.ModuleType("Some_Renamed_RefMod_Folder.nodes")
    fake._ref_blocks = lambda *a, **k: []
    fake.MiniMaxH3RefModApply = object
    fake.MiniMaxH3RefModsLoader = object
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    assert resolve_refmod_pack() is fake
    monkeypatch.delitem(sys.modules, fake.__name__)
    decoy = types.ModuleType("decoy.nodes")
    decoy._ref_blocks = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, decoy.__name__, decoy)
    assert resolve_refmod_pack() is None


# ------------------------------------------------------------- model attach


def test_attach_injects_into_every_cond_and_restores_after_sampling():
    model = _Model({"other": {"sage": [lambda: None]}})
    blocks = [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "L", "refmod": True}]
    attached = attach_refmod_blocks(model, blocks)
    assert attached is not model
    assert model.wrappers == {"other": {"sage": model.wrappers["other"]["sage"]}}
    group = refmod_bridge._outer_sample_key()
    wrappers = attached.wrappers[group][BRIDGE_WRAPPER_KEY]
    assert len(wrappers) == 1

    guider = types.SimpleNamespace(conds={
        "positive": [{"cross_attn": "c", "minimax_refs": [{"kind": "video", "latent": "ctx"}]},
                     {"cross_attn": "d"}],
    })
    original = guider.conds
    executor = _Executor(guider)
    assert wrappers[0](executor, "noise", "latent") == "sampled"
    assert executor.seen["positive"][0]["minimax_refs"] == [
        {"kind": "video", "latent": "ctx"},
        {"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "L", "refmod": True},
    ]
    assert executor.seen["positive"][1]["minimax_refs"] == [
        {"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "L", "refmod": True},
    ]
    assert guider.conds is original
    assert "minimax_refs" not in original["positive"][1]

    # Exceptions inside sampling must also restore the guider conditioning.
    class _Failing:
        class_obj = guider
        def __call__(self, *a, **k):
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError):
        wrappers[0](_Failing(), "noise")
    assert guider.conds is original


def test_attach_with_no_blocks_or_reattach_is_idempotent():
    model = _Model()
    passthrough = attach_refmod_blocks(model, [])
    assert passthrough.wrappers == {}
    blocks = [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": "L"}]
    once = attach_refmod_blocks(model, blocks)
    twice = attach_refmod_blocks(once, blocks)
    group = refmod_bridge._outer_sample_key()
    assert len(twice.wrappers[group][BRIDGE_WRAPPER_KEY]) == 1
    assert attach_refmod_blocks(once, []).wrappers.get(group, {}) == {}


def test_attach_requires_the_model_patcher_api():
    with pytest.raises(RefModBridgeError):
        attach_refmod_blocks(object(), [{"kind": "image"}])


# ----------------------------------------------------------------- sampler


def test_v38_schema_exposes_refmods_socket_and_apply_settings_with_help():
    schema = H3ContinuumSamplerV38.INPUT_TYPES()
    assert schema["optional"]["refmods"][0] == "H3_REF_MODS"
    assert schema["optional"]["refmods"][1]["tooltip"]
    required = schema["required"]
    assert list(required)[-len(REFMOD_WIDGETS):] == list(REFMOD_WIDGETS)
    for name in REFMOD_WIDGETS:
        definition = required[name]
        assert definition[1]["advanced"] is True
        assert definition[1]["tooltip"].strip()
        assert definition[1]["display_name"].startswith("RefMod")
    assert required["refmod_curve_direction"][0] == list(CURVE_DIRECTIONS)
    assert required["refmod_curve_shape"][0] == list(CURVE_SHAPES)
    assert required["refmod_scramble_mode"][0] == list(SCRAMBLE_MODES)
    assert required["refmod_retention"][1]["default"] == 1.0
    assert required["refmod_scramble_seed"][1]["default"] == -1
    assert required["refmod_override"][1]["default"] is False


def test_v38_run_without_refmods_is_a_pure_passthrough(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return ("video", "audio", {"plan": True}, "report")

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    model = _Model()
    outputs = H3ContinuumSamplerV38().run(
        model=model, chunk_seconds=5.0, diagnostics="Off",
        refmod_retention=0.2, refmod_scramble_seed=5,
    )
    assert captured["model"] is model
    assert not any(key.startswith("refmod") for key in captured)
    assert outputs[3] == "report"


def test_v38_run_with_refmods_attaches_a_scoped_clone_and_reports(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return ("video", "audio", {"plan": True}, "report")

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    monkeypatch.setattr(refmod_bridge, "resolve_refmod_pack", lambda: None)
    model = _Model()
    a, b = _Mod("hero"), _Mod("style")
    outputs = H3ContinuumSamplerV38().run(
        model=model, chunk_seconds=5.0, diagnostics="Off",
        refmods=[(a, 1.0), (b, 0.5)],
        refmod_retention=0.7,
    )
    attached = captured["model"]
    assert attached is not model
    group = refmod_bridge._outer_sample_key()
    assert BRIDGE_WRAPPER_KEY in attached.wrappers[group]
    assert model.wrappers == {}
    assert "refmods" not in captured
    status = outputs[3]
    assert status.startswith("report\nRefMod Bridge")
    assert "hero@1.00" in status and "style@0.50" in status
    assert "retention=0.70" in status
    assert "injected into every chunk" in status


def test_v38_run_with_empty_bundle_keeps_model_and_explains(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return ("video", "audio", {"plan": True}, "report")

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    model = _Model()
    outputs = H3ContinuumSamplerV38().run(
        model=model, chunk_seconds=5.0, diagnostics="Off", refmods=[],
    )
    assert captured["model"] is model
    assert "Bundle connected but empty" in outputs[3]


def test_v38_run_reports_refmods_before_resolution_and_reliability(monkeypatch):
    def fake_v37_run(self, **kwargs):
        return ("video", "audio", {"plan": True}, "report", "refine")

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    monkeypatch.setattr(refmod_bridge, "resolve_refmod_pack", lambda: None)
    outputs = H3ContinuumSamplerV38().run(
        model=_Model(), chunk_seconds=5.0, diagnostics="Off",
        size_source="Manual", width=480, height=640,
        refmods=[(_Mod("only"), 1.0)],
    )
    assert outputs[4] == "refine"
    lines = outputs[3].splitlines()
    assert lines[0] == "report"
    assert lines[1] == "RefMod Bridge"
    assert any(line.startswith("Resolution:") for line in lines)


def test_format_status_mentions_scramble_and_notes():
    summary = {
        "mod_count": 2,
        "mods": [("a", 1.0, 16), ("b", 0.5, 0)],
        "block_count": 2,
        "token_count": 16,
        "helper": "continuum fallback",
        "settings": {
            "retention": 0.7, "retention_source": "sampler widget",
            "curve_direction": "constant", "curve_shape": "linear", "curve_value": 1.0,
            "curve_source": "sampler widgets",
            "scramble_seed": 4, "scramble_mode": "subset", "scramble_keep": 1,
        },
        "notes": ["something"],
    }
    text = format_refmod_status(summary)
    assert "scramble=seed 4, subset keep 1" in text
    assert "Note: something" in text
    assert "b@0.50" in text and "(0 tok)" not in text


def test_frontend_wires_refmod_widgets_into_the_advanced_disclosure():
    source = (ROOT / "web" / "project_id.js").read_text(encoding="utf-8")
    assert "const REFMOD_WIDGETS = Object.freeze([" in source
    for name in REFMOD_WIDGETS:
        assert f'"{name}",' in source
        assert f"    {name}: (" in source
    assert 'activeLinkedInput(node, ["refmods"])' in source
    assert "setWidgetVisible(findWidget(node, name), advanced && refmodsConnected);" in source


def test_status_reports_block_tensor_stats_and_flags_non_finite():
    import torch

    class _TensorMod(_Mod):
        def __init__(self, name, value):
            super().__init__(name)
            self._value = value

        def ref_block(self, strength=1.0, curve=None):
            block = super().ref_block(strength, curve)
            block["latent"] = self._value
            return block

    good = _TensorMod("good", torch.full((1, 24, 1, 8, 8), 0.5))
    bad = _TensorMod("bad", torch.full((1, 24, 1, 8, 8), float("nan")))
    _, summary = build_refmod_blocks([(good, 1.0), (bad, 1.0)], normalize_settings(), pack=None)
    stats = summary["block_stats"]
    assert stats[0]["latent"]["finite"] is True
    assert stats[0]["latent"]["shape"] == (1, 24, 1, 8, 8)
    assert stats[1]["latent"]["finite"] is False
    text = format_refmod_status(summary)
    assert "Block 1: kind=image; t=1; hw=8x8; latent=(1, 24, 1, 8, 8) float32@cpu finite=True absmax=0.500" in text
    assert "WARNING: reference block(s) 2 contain NaN/Inf" in text
