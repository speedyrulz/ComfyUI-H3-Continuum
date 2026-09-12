from __future__ import annotations

import inspect

import pytest
import torch

from ComfyUI_H3_Continuum_Join import reference, reference_video
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes, reliability_v38
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)
from ComfyUI_H3_Continuum_Join.v3.reliability_v38 import (
    V38Diagnostics,
    append_v38_status,
    build_frame_alignment_diagnostics,
    collect_core_compatibility_advisory,
    estimate_visual_conditioning_load,
    format_v38_diagnostics,
    validate_audio_latent_lengths,
)


def _group(
    frames: int,
    *,
    trim: int = 0,
    video_t: int,
    audio_t: int,
    logical=(1,),
    terminal=False,
):
    return {
        "sequence_index": 1,
        "chunk_index": 1,
        "total_frames": frames,
        "trim_frames": trim,
        "net_frames": frames - trim,
        "context_frames": trim,
        "expected_video_latent_t": video_t,
        "expected_audio_latent_t": audio_t,
        "logical_chunk_indices": list(logical),
        "terminal_merged": terminal,
    }


def _plan(*groups, terminal=False):
    if terminal:
        return {
            "chunks": [
                _group(124, video_t=37, audio_t=207),
                _group(141, trim=22, video_t=42, audio_t=235),
            ],
            "decode_groups": list(groups),
        }
    return {"chunks": list(groups)}


def _video_latent(t: int):
    return {"samples": torch.zeros(1, 24, t, 2, 2)}


def _audio_latent(t: int, *, channels=32, stereo=2):
    return {"samples": torch.zeros(1, channels, stereo, t)}


def test_current_v38_public_schema_is_preserved_exactly():
    v37 = H3ContinuumSamplerV37.INPUT_TYPES()
    v38 = H3ContinuumSamplerV38.INPUT_TYPES()
    expected_required = []
    for name in v37["required"]:
        if name == "width":
            expected_required.extend(("aspect", "preset", "custom_mp"))
        elif name != "height":
            expected_required.append(name)
    expected_required.extend(
        (
            "generation_mode",
            "review_action",
            "take_group",
            "take_revision_id",
            "take_action",
            "size_source",
            "width",
            "height",
        )
    )
    # RefMod (ComfyUI-MiniMaxH3Mod) controls are appended last so every
    # earlier V3.8 widget index is unchanged.
    expected_required.extend(
        (
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
    )
    assert tuple(v38["required"]) == tuple(expected_required)
    assert list(v38["optional"])[:-2] == list(v37["optional"])
    assert list(v38["optional"])[-2:] == ["audio_references", "refmods"]
    assert v38["optional"]["audio_references"][0] == (
        "H3_CONTINUUM_AUDIO_REFERENCES"
    )
    assert v38.get("hidden") == v37.get("hidden")
    # The review/size widgets stay the last non-RefMod controls; `refmod_*`
    # widgets are appended after them.
    assert tuple(name for name in v38["required"] if not name.startswith("refmod_"))[-8:] == (
        "generation_mode",
        "review_action",
        "take_group",
        "take_revision_id",
        "take_action",
        "size_source",
        "width",
        "height",
    )


def test_reference_preprocessing_uses_the_new_shared_pure_size_resolver(monkeypatch):
    calls = []

    def fake_resolver(source_width, source_height, **kwargs):
        calls.append((source_width, source_height, kwargs))
        return source_width, source_height

    monkeypatch.setattr(reference, "resolve_reference_image_size", fake_resolver)
    image = torch.zeros(1, 64, 96, 3)
    assets = reference.prepare_reference_assets(
        reference_image_1=image,
        reference_image_2=None,
        output_width=96,
        output_height=64,
        size_mode=reference.REFERENCE_SIZE_MATCH_OUTPUT,
    )
    assert assets is not None
    assert assets.images[0].shape == image.shape
    assert calls == [
        (
            96,
            64,
            {
                "output_width": 96,
                "output_height": 64,
                "size_mode": reference.REFERENCE_SIZE_MATCH_OUTPUT,
            },
        )
    ]


def test_reference_video_preprocessing_uses_shared_frame_and_size_resolvers(
    monkeypatch,
):
    assert reference_video._resolved_size is reference_video.resolve_reference_video_size
    assert (
        reference_video._resolve_reference_frame_count
        is reference_video.resolve_reference_video_frame_count
    )
    frame_calls = []
    size_calls = []

    def fake_frames(source_frames, target_frames):
        frame_calls.append((source_frames, target_frames))
        return 5

    def fake_size(source_width, source_height, **kwargs):
        size_calls.append((source_width, source_height, kwargs))
        return source_width, source_height

    monkeypatch.setattr(
        reference_video,
        "resolve_reference_video_frame_count",
        fake_frames,
    )
    monkeypatch.setattr(
        reference_video,
        "resolve_reference_video_size",
        fake_size,
    )
    source = reference_video.prepare_reference_video_source(
        torch.zeros(5, 32, 64, 3),
        target_frames=124,
        output_width=64,
        output_height=32,
        size_mode=reference_video.REFERENCE_VIDEO_SIZE_EFFICIENT,
    )
    assert source is not None
    assert frame_calls == [(5, 124)]
    assert size_calls == [
        (
            64,
            32,
            {
                "output_width": 64,
                "output_height": 32,
                "size_mode": reference_video.REFERENCE_VIDEO_SIZE_EFFICIENT,
            },
        )
    ]


@pytest.mark.parametrize(
    ("source_frames", "expected_action"),
    (
        (120, "120 -> 124 (final-frame pad"),
        (124, "124 -> 124 (unchanged"),
        (180, "180 -> 124 (trimmed to one-chunk H3 limit"),
    ),
)
def test_video_guide_alignment_reports_existing_pad_unchanged_and_cap(
    source_frames,
    expected_action,
):
    section = build_frame_alignment_diagnostics(
        video_latents=[_video_latent(37)],
        assembly_plan=_plan(_group(124, video_t=37, audio_t=207)),
        video_guide=torch.zeros(source_frames, 32, 64, 3),
        chunk_seconds=5.0,
    )
    assert expected_action in "\n".join(section.detailed)
    if source_frames == 124:
        assert not section.basic
    else:
        assert expected_action in "\n".join(section.basic)


def test_frame_grid_uses_terminal_physical_group_and_advises_on_actual_t_mismatch():
    terminal = _group(
        260,
        trim=22,
        video_t=77,
        audio_t=433,
        logical=(2, 3),
        terminal=True,
    )
    passed = build_frame_alignment_diagnostics(
        video_latents=[_video_latent(77)],
        assembly_plan=_plan(terminal, terminal=True),
        chunk_seconds=5.0,
    )
    assert not passed.basic
    assert "frames=260" in passed.detailed[0]
    assert "video_t=77" in passed.detailed[0]
    assert passed.detailed[0].endswith("PASS.")

    failed = build_frame_alignment_diagnostics(
        video_latents=[_video_latent(76)],
        assembly_plan=_plan(terminal, terminal=True),
        chunk_seconds=5.0,
    )
    assert "actual T differs" in "\n".join(failed.basic)
    assert failed.detailed[0].endswith("ADVISORY.")


def test_frame_grid_advises_on_physical_output_count_mismatch():
    section = build_frame_alignment_diagnostics(
        video_latents=[],
        assembly_plan=_plan(_group(124, video_t=37, audio_t=207)),
        chunk_seconds=5.0,
    )
    assert "output count 0 does not match group count 1" in "\n".join(section.basic)


def test_audio_validation_passes_normal_and_terminal_physical_groups():
    first = _group(124, video_t=37, audio_t=207)
    terminal = _group(
        260,
        trim=22,
        video_t=77,
        audio_t=433,
        logical=(2, 3),
        terminal=True,
    )
    section = validate_audio_latent_lengths(
        audio_latents=[_audio_latent(207), _audio_latent(433)],
        assembly_plan=_plan(first, terminal, terminal=True),
    )
    assert not section.basic
    assert len(section.detailed) == 2
    assert all(line.endswith("PASS.") for line in section.detailed)
    assert "phase=0.333333" in section.detailed[0]
    assert "phase=-0.333333" in section.detailed[1]


@pytest.mark.parametrize(
    ("latent", "expected"),
    (
        ({}, "samples are missing"),
        ({"samples": "not-a-tensor"}, "not a Tensor"),
        ({"samples": torch.zeros(1, 32, 207)}, "rank is 3"),
        (_audio_latent(207, channels=31), "channels=31"),
        (_audio_latent(207, stereo=1), "stereo=1"),
        (_audio_latent(0), "T must be positive"),
    ),
)
def test_audio_validation_reports_shape_advisories(latent, expected):
    section = validate_audio_latent_lengths(
        audio_latents=[latent],
        assembly_plan=_plan(_group(124, video_t=37, audio_t=207)),
    )
    assert expected in "\n".join(section.basic)


def test_audio_validation_reports_metadata_delta_phase_and_count():
    section = validate_audio_latent_lengths(
        audio_latents=[_audio_latent(206)],
        assembly_plan=_plan(
            _group(124, video_t=37, audio_t=207),
            _group(141, trim=22, video_t=42, audio_t=235),
        ),
    )
    text = "\n".join(section.basic)
    assert "output count 1 does not match group count 2" in text
    assert "metadata=207 actual=206 delta=-1" in text
    assert "phase=-0.666667" in text


def test_audio_validation_does_not_copy_or_modify_tensor():
    samples = torch.arange(1 * 32 * 2 * 207, dtype=torch.float32).reshape(
        1, 32, 2, 207
    )
    before = samples.clone()
    pointer = samples.data_ptr()
    latents = [{"samples": samples}]
    section = validate_audio_latent_lengths(
        audio_latents=latents,
        assembly_plan=_plan(_group(124, video_t=37, audio_t=207)),
    )
    assert not section.basic
    assert latents[0]["samples"] is samples
    assert samples.data_ptr() == pointer
    assert torch.equal(samples, before)
    source = inspect.getsource(validate_audio_latent_lengths)
    for forbidden in (".clone(", ".cpu(", ".to(", ".numpy(", "contiguous("):
        assert forbidden not in source


def test_core_advisory_basic_detailed_deduplicates_and_never_gates():
    compatible = collect_core_compatibility_advisory(lambda: [])
    assert compatible.basic == ()
    assert "native MiniMax H3 contracts detected" in compatible.detailed[0]

    issues = collect_core_compatibility_advisory(
        lambda: ["PackedLayout changed", "PackedLayout changed", "rows missing"]
    )
    assert "2 native H3 contract difference" in issues.basic[0]
    assert len(issues.detailed) == 2

    unavailable = collect_core_compatibility_advisory(
        lambda: (_ for _ in ()).throw(RuntimeError("probe failed"))
    )
    assert "generation result was preserved" in unavailable.basic[0]
    assert "RuntimeError: probe failed" in unavailable.detailed[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.749999, "Low"),
        (0.75, "Medium"),
        (1.749999, "Medium"),
        (1.75, "High"),
    ),
)
def test_visual_load_classification_boundaries(value, expected):
    assert reliability_v38._visual_load_class(value) == expected


def test_visual_load_uses_existing_efficient_geometry_and_temporal_ratio():
    section = estimate_visual_conditioning_load(
        output_width=1344,
        output_height=768,
        reference_images=(),
        reference_size=reference.REFERENCE_SIZE_MATCH_OUTPUT,
        video_guide=torch.zeros(120, 1080, 1920, 3),
        video_guide_size=reference_video.REFERENCE_VIDEO_SIZE_EFFICIENT,
        chunk_seconds=5.0,
        still_guide_active=False,
    )
    assert section.basic == (
        "Visual conditioning load: Low (peak per physical group).",
    )
    assert "total=0.387 class=Low" in section.detailed[0]
    assert "Video Guide=832x480/124f=0.387" in section.detailed[0]


def test_visual_load_adds_reference_images_but_excludes_still_guide():
    reference_image = torch.zeros(1, 768, 1024, 3)
    section = estimate_visual_conditioning_load(
        output_width=1024,
        output_height=768,
        reference_images=(reference_image, reference_image, reference_image),
        reference_size=reference.REFERENCE_SIZE_MATCH_OUTPUT,
        video_guide=None,
        chunk_seconds=5.0,
        still_guide_active=True,
    )
    assert "High" in section.basic[0]
    assert "Still Guide active" in section.basic[0]
    assert "total=3.000" in section.detailed[0]
    assert "Still Guide=active" in section.detailed[0]
    assert "Ref1=1024x768=1.000" in section.detailed[0]
    assert "Ref3=1024x768=1.000" in section.detailed[0]


def test_still_guide_only_is_reported_without_numeric_total():
    section = estimate_visual_conditioning_load(
        output_width=1024,
        output_height=768,
        reference_images=(),
        reference_size=reference.REFERENCE_SIZE_MATCH_OUTPUT,
        video_guide=None,
        chunk_seconds=5.0,
        still_guide_active=True,
    )
    assert section.basic == ("Still Guide: active.",)
    assert "excluded from numeric visual load" in section.detailed[0]


def test_visual_load_inspects_shapes_only_and_has_no_chunk_or_storage_input():
    signature = inspect.signature(estimate_visual_conditioning_load)
    assert "chunks" not in signature.parameters
    assert "run_storage" not in signature.parameters
    source = inspect.getsource(estimate_visual_conditioning_load)
    for forbidden in (
        ".clone(",
        ".cpu(",
        ".to(",
        ".numpy(",
        "encode(",
        "_tensor_hash",
        "common_upscale",
    ):
        assert forbidden not in source


def test_basic_detailed_and_off_status_formatting():
    diagnostics = V38Diagnostics(
        basic=("Basic issue.",),
        detailed=("Detailed PASS.",),
    )
    assert format_v38_diagnostics(diagnostics, mode="Basic") == (
        "V3.8 Reliability\nBasic issue."
    )
    assert format_v38_diagnostics(diagnostics, mode="Detailed Report") == (
        "V3.8 Reliability\nDetailed PASS."
    )
    assert format_v38_diagnostics(diagnostics, mode="Off") == ""
    assert append_v38_status("Original", diagnostics, mode="Off") == "Original"
    assert append_v38_status("Original", diagnostics, mode="Basic") == (
        "Original\nV3.8 Reliability\nBasic issue."
    )


def test_v38_wrapper_preserves_every_non_status_output_identity(monkeypatch):
    video = [_video_latent(37)]
    audio = [_audio_latent(207)]
    plan = _plan(_group(124, video_t=37, audio_t=207))
    driving_audio = object()
    refine_context = object()

    def fake_v37_run(self, **kwargs):
        return video, audio, plan, "Original status", driving_audio, refine_context

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    monkeypatch.setattr(
        driving_nodes,
        "build_v38_diagnostics",
        lambda **kwargs: V38Diagnostics(("Basic diagnostic.",), ("Detailed.",)),
    )
    outputs = H3ContinuumSamplerV38().run(
        chunk_seconds=5.0,
        diagnostics="Basic",
    )
    assert outputs[0] is video
    assert outputs[1] is audio
    assert outputs[2] is plan
    assert outputs[4] is driving_audio
    assert outputs[5] is refine_context
    assert outputs[3] == (
        "Original status\nV3.8 Reliability\nBasic diagnostic."
    )


def test_v38_wrapper_preserves_generation_when_diagnostics_raise(monkeypatch):
    outputs = (
        [_video_latent(37)],
        [_audio_latent(207)],
        _plan(_group(124, video_t=37, audio_t=207)),
        "Original status",
        object(),
    )

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", lambda self, **kwargs: outputs)
    monkeypatch.setattr(
        driving_nodes,
        "build_v38_diagnostics",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("diagnostic failed")),
    )
    result = H3ContinuumSamplerV38().run(
        chunk_seconds=5.0,
        diagnostics="Detailed Report",
    )
    assert result[0] is outputs[0]
    assert result[1] is outputs[1]
    assert result[2] is outputs[2]
    assert result[4] is outputs[4]
    assert "generation result was preserved" in result[3]
    assert "RuntimeError: diagnostic failed" in result[3]


def test_v38_wrapper_keeps_non_tuple_test_double_unchanged(monkeypatch):
    monkeypatch.setattr(H3ContinuumSamplerV37, "run", lambda self, **kwargs: "ok")
    assert H3ContinuumSamplerV38().run() == "ok"


def test_v38_wrapper_diagnostics_off_skips_all_reliability_work(monkeypatch):
    outputs = (
        [_video_latent(37)],
        [_audio_latent(207)],
        _plan(_group(124, video_t=37, audio_t=207)),
        "Original status",
    )
    monkeypatch.setattr(H3ContinuumSamplerV37, "run", lambda self, **kwargs: outputs)

    def fail_if_called(**kwargs):
        raise AssertionError("diagnostics must not run when Report Detail is Off")

    monkeypatch.setattr(driving_nodes, "build_v38_diagnostics", fail_if_called)
    result = H3ContinuumSamplerV38().run(
        chunk_seconds=5.0,
        diagnostics="Off",
    )
    assert result is outputs
