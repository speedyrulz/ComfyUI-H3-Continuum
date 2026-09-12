from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from ComfyUI_H3_Continuum_Join.v3.easy_nodes import resolve_easy_resolution
from ComfyUI_H3_Continuum_Join.v3.resolution import (
    H3_ASPECT_AUTO,
    H3_ASPECT_LANDSCAPE,
    H3_ASPECT_OPTIONS,
    H3_ASPECT_PORTRAIT,
    H3_ASPECT_SQUARE,
    H3_CANVAS_MAX,
    H3_CANVAS_MIN,
    H3_CANVAS_MULTIPLE,
    H3_CUSTOM_MP_MAX,
    H3_CUSTOM_MP_MIN,
    H3_MANUAL_HEIGHT_DEFAULT,
    H3_MANUAL_WIDTH_DEFAULT,
    H3_PRESET_BALANCED,
    H3_PRESET_CUSTOM,
    H3_PRESET_DRAFT,
    H3_PRESET_NATIVE,
    H3_PRESET_OPTIONS,
    H3_SIZE_SOURCE_FIRST_IMAGE,
    H3_SIZE_SOURCE_LEGACY,
    H3_SIZE_SOURCE_MANUAL,
    H3_SIZE_SOURCE_OPTIONS,
    resolve_h3_resolution,
    resolve_h3_size_source,
)


ROOT = Path(__file__).resolve().parents[1]


class _Image:
    def __init__(self, height: int, width: int):
        self.shape = (1, height, width, 3)


@pytest.mark.parametrize("aspect", H3_ASPECT_OPTIONS)
@pytest.mark.parametrize("preset", H3_PRESET_OPTIONS)
def test_main_and_easy_share_exact_resolution_results(aspect, preset):
    first = _Image(900, 1200)
    custom_mp = 1.23
    main = resolve_h3_resolution(
        aspect=aspect,
        preset=preset,
        custom_mp=custom_mp,
        first_frame=first,
    )
    easy = resolve_easy_resolution(
        aspect=aspect,
        preset=preset,
        custom_mp=custom_mp,
        first_frame=first,
    )
    assert main == easy
    assert main.width % 32 == 0
    assert main.height % 32 == 0


@pytest.mark.parametrize(
    ("first", "orientation"),
    (
        (_Image(720, 1280), "landscape"),
        (_Image(1280, 720), "portrait"),
        (_Image(900, 900), "square"),
    ),
)
def test_auto_uses_first_image_geometry(first, orientation):
    plan = resolve_h3_resolution(
        aspect=H3_ASPECT_AUTO,
        preset=H3_PRESET_DRAFT,
        first_frame=first,
    )
    assert plan.aspect_source == "First Image"
    if orientation == "landscape":
        assert plan.width > plan.height
    elif orientation == "portrait":
        assert plan.width < plan.height
    else:
        assert plan.width == plan.height


def test_auto_without_first_image_preserves_easy_landscape_fallback():
    auto = resolve_h3_resolution(
        aspect=H3_ASPECT_AUTO,
        preset=H3_PRESET_DRAFT,
    )
    landscape = resolve_h3_resolution(
        aspect=H3_ASPECT_LANDSCAPE,
        preset=H3_PRESET_DRAFT,
    )
    assert (auto.width, auto.height) == (landscape.width, landscape.height)
    assert auto.aspect_source == "Landscape 16:9 fallback"


@pytest.mark.parametrize(
    ("preset", "target_mp", "tolerance"),
    (
        (H3_PRESET_DRAFT, 0.30, 0.04),
        (H3_PRESET_BALANCED, 0.60, 0.05),
    ),
)
def test_decimal_mp_presets_are_near_their_targets(preset, target_mp, tolerance):
    for aspect in (
        H3_ASPECT_LANDSCAPE,
        H3_ASPECT_PORTRAIT,
        H3_ASPECT_SQUARE,
    ):
        plan = resolve_h3_resolution(aspect=aspect, preset=preset)
        assert abs(plan.actual_mp - target_mp) <= tolerance


@pytest.mark.parametrize("custom_mp", (H3_CUSTOM_MP_MIN, 0.37, 2.50, H3_CUSTOM_MP_MAX))
def test_custom_mp_range_and_alignment(custom_mp):
    plan = resolve_h3_resolution(
        aspect=H3_ASPECT_PORTRAIT,
        preset=H3_PRESET_CUSTOM,
        custom_mp=custom_mp,
    )
    assert plan.width % 32 == plan.height % 32 == 0
    assert plan.preset == H3_PRESET_CUSTOM


@pytest.mark.parametrize("custom_mp", (H3_CUSTOM_MP_MIN - 0.01, H3_CUSTOM_MP_MAX + 0.01, True))
def test_custom_mp_validation_matches_easy(custom_mp):
    with pytest.raises((TypeError, ValueError)):
        resolve_h3_resolution(
            aspect=H3_ASPECT_SQUARE,
            preset=H3_PRESET_CUSTOM,
            custom_mp=custom_mp,
        )


def test_invalid_aspect_and_preset_preserve_existing_easy_fallback_contract():
    invalid_aspect = resolve_h3_resolution(
        aspect="invalid",
        preset=H3_PRESET_DRAFT,
    )
    landscape = resolve_h3_resolution(
        aspect=H3_ASPECT_LANDSCAPE,
        preset=H3_PRESET_DRAFT,
    )
    invalid_preset = resolve_h3_resolution(
        aspect=H3_ASPECT_SQUARE,
        preset="invalid",
        custom_mp=0.42,
    )
    custom = resolve_h3_resolution(
        aspect=H3_ASPECT_SQUARE,
        preset=H3_PRESET_CUSTOM,
        custom_mp=0.42,
    )
    assert (invalid_aspect.width, invalid_aspect.height) == (
        landscape.width,
        landscape.height,
    )
    assert invalid_preset == custom


def test_v38_schema_keeps_legacy_aspect_and_appends_size_source_contract():
    v37 = H3ContinuumSamplerV37.INPUT_TYPES()
    v38 = H3ContinuumSamplerV38.INPUT_TYPES()
    assert "width" in v37["required"]
    assert "height" in v37["required"]
    required_names = list(v38["required"])
    # RefMod controls are appended after the size contract; the size trio must
    # still be the last non-RefMod widgets so every earlier index is stable.
    non_refmod = [name for name in required_names if not name.startswith("refmod_")]
    assert non_refmod[-3:] == ["size_source", "width", "height"]
    assert required_names.index("height") < required_names.index("refmod_retention")
    assert tuple(name for name in v38["required"] if name in {"aspect", "preset", "custom_mp"}) == (
        "aspect",
        "preset",
        "custom_mp",
    )
    assert v38["required"]["aspect"][0] == H3_ASPECT_OPTIONS
    assert v38["required"]["preset"][0] == H3_PRESET_OPTIONS
    assert v38["required"]["custom_mp"][1]["min"] == H3_CUSTOM_MP_MIN
    assert v38["required"]["custom_mp"][1]["max"] == H3_CUSTOM_MP_MAX
    assert v38["required"]["aspect"][1]["advanced"] is True
    assert v38["required"]["size_source"][0] == H3_SIZE_SOURCE_OPTIONS
    assert v38["required"]["size_source"][1]["default"] == H3_SIZE_SOURCE_LEGACY
    assert v38["required"]["width"][1] == {
        "default": H3_MANUAL_WIDTH_DEFAULT,
        "min": H3_CANVAS_MIN,
        "max": H3_CANVAS_MAX,
        "step": H3_CANVAS_MULTIPLE,
        "display_name": "Width",
        "tooltip": (
            "Exact output width in Manual mode. Use a multiple of 32. Manual mode is "
            "the normal choice for T2VA or workflows without a First Image."
        ),
    }
    # V3.8 appends `audio_references` and then the RefMod `refmods` socket.
    assert list(v38["optional"])[:-2] == list(v37["optional"])
    assert list(v38["optional"])[-2:] == ["audio_references", "refmods"]
    assert list(v38["optional"])[-2] == "audio_references"
    assert v38["optional"]["audio_references"][0] == (
        "H3_CONTINUUM_AUDIO_REFERENCES"
    )
    assert v38.get("hidden") == v37.get("hidden")


def test_v38_registration_is_the_public_sampler_surface():
    assert NODE_CLASS_MAPPINGS["H3ContinuumSamplerV38"] is H3ContinuumSamplerV38
    assert NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumSamplerV38"] == (
        "H3 Continuum Sampler V3.8"
    )
    assert root_nodes.NODE_CLASS_MAPPINGS["H3ContinuumSamplerV38"] is (
        H3ContinuumSamplerV38
    )
    assert H3ContinuumSamplerV38.DEPRECATED is False
    assert "H3ContinuumSamplerV37" not in root_nodes.NODE_CLASS_MAPPINGS


def test_v38_facade_passes_only_resolved_geometry_to_v37(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    result = H3ContinuumSamplerV38().run(
        size_source=H3_SIZE_SOURCE_FIRST_IMAGE,
        aspect=H3_ASPECT_AUTO,
        preset=H3_PRESET_NATIVE,
        custom_mp=0.30,
        first_frame=_Image(1200, 900),
        project_id="stable-project",
    )
    expected = resolve_h3_size_source(
        size_source=H3_SIZE_SOURCE_FIRST_IMAGE,
        preset=H3_PRESET_NATIVE,
        first_frame=_Image(1200, 900),
    )
    assert result == "ok"
    assert (captured["width"], captured["height"]) == (
        expected.width,
        expected.height,
    )
    assert captured["project_id"] == "stable-project"
    assert "aspect" not in captured
    assert "preset" not in captured
    assert "custom_mp" not in captured
    assert "size_source" not in captured


def test_run_storage_resolution_identity_uses_resolved_width_and_height(monkeypatch):
    captured = []

    def fake_v37_run(self, **kwargs):
        captured.append(kwargs)
        return "ok"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    node = H3ContinuumSamplerV38()
    node.run(
        size_source=H3_SIZE_SOURCE_MANUAL,
        width=768,
        height=768,
        preset=H3_PRESET_BALANCED,
    )
    node.run(
        size_source=H3_SIZE_SOURCE_MANUAL,
        width=768,
        height=768,
        preset=H3_PRESET_CUSTOM,
        custom_mp=0.589824,
    )
    assert [(value["width"], value["height"]) for value in captured] == [
        (768, 768),
        (768, 768),
    ]
    assert all("preset" not in value for value in captured)


@pytest.mark.parametrize(
    ("height", "width"),
    ((720, 1280), (1280, 720), (900, 900), (800, 1200), (1200, 800)),
)
@pytest.mark.parametrize(
    "preset",
    (H3_PRESET_DRAFT, H3_PRESET_BALANCED, H3_PRESET_NATIVE),
)
def test_first_image_size_source_preserves_unusual_aspects(height, width, preset):
    image = _Image(height, width)
    plan = resolve_h3_size_source(
        size_source=H3_SIZE_SOURCE_FIRST_IMAGE,
        preset=preset,
        first_frame=image,
    )
    legacy = resolve_h3_resolution(
        aspect=H3_ASPECT_AUTO,
        preset=preset,
        first_frame=image,
    )
    assert plan == legacy
    assert plan.width % 32 == plan.height % 32 == 0
    assert abs(plan.aspect_ratio - width / height) < 1e-12


def test_first_image_missing_fails_open_to_visible_manual_canvas():
    plan = resolve_h3_size_source(
        size_source=H3_SIZE_SOURCE_FIRST_IMAGE,
        width=704,
        height=1024,
        preset=H3_PRESET_BALANCED,
    )
    assert (plan.width, plan.height) == (704, 1024)
    assert plan.aspect_source == "Manual Width / Height"
    assert "First Image is unavailable" in plan.warnings[0]


def test_v38_status_reports_exact_resolved_canvas_and_missing_image_fallback(monkeypatch):
    def fake_v37_run(self, **kwargs):
        return ("video", "audio", {}, "Base status", None, None)

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    outputs = H3ContinuumSamplerV38().run(
        size_source=H3_SIZE_SOURCE_FIRST_IMAGE,
        width=704,
        height=1024,
        diagnostics="Off",
    )
    assert "Resolution: 704 x 1024 (0.72 MP)" in outputs[3]
    assert "source=Manual Width / Height" in outputs[3]
    assert "First Image is unavailable" in outputs[3]


@pytest.mark.parametrize(
    ("width", "height"),
    ((1024, 576), (576, 1024), (768, 768), (704, 1024), (960, 640)),
)
def test_manual_size_source_uses_exact_aligned_dimensions(width, height):
    plan = resolve_h3_size_source(
        size_source=H3_SIZE_SOURCE_MANUAL,
        width=width,
        height=height,
        preset=H3_PRESET_DRAFT,
    )
    assert (plan.width, plan.height) == (width, height)
    assert plan.target_mp is None
    assert plan.preset == "Manual"


@pytest.mark.parametrize("width,height", ((703, 1024), (704, 1023), (0, 32), (32, 16416)))
def test_manual_size_source_rejects_unaligned_or_out_of_range_canvas(width, height):
    with pytest.raises(ValueError):
        resolve_h3_size_source(
            size_source=H3_SIZE_SOURCE_MANUAL,
            width=width,
            height=height,
        )


@pytest.mark.parametrize("aspect", H3_ASPECT_OPTIONS)
def test_legacy_size_source_remains_bit_exact_for_old_api_prompts(aspect):
    legacy = resolve_h3_size_source(
        size_source=H3_SIZE_SOURCE_LEGACY,
        aspect=aspect,
        preset=H3_PRESET_BALANCED,
        first_frame=_Image(900, 1200),
    )
    expected = resolve_h3_resolution(
        aspect=aspect,
        preset=H3_PRESET_BALANCED,
        first_frame=_Image(900, 1200),
    )
    assert legacy == expected


def test_project_frontend_applies_v37_controls_and_custom_mp_visibility_to_v38():
    source = (ROOT / "web" / "project_id.js").read_text(encoding="utf-8")
    assert 'const V38_NODE_CLASS = "H3ContinuumSamplerV38";' in source
    assert "isV35 || isV36 || isV37 || isV38" in source
    assert 'node.comfyClass === V38_NODE_CLASS' in source
    assert 'presetWidget?.value === CUSTOM_RESOLUTION_PRESET' in source
    assert 'attachRefresh(presetWidget, "__h3ContinuumResolutionPresetCallback", refresh);' in source
    assert 'const SIZE_SOURCE_WIDGET = "size_source";' in source
    assert "migrateLegacyAspect();" in source
    assert "sizeSourceWidget.options.values = [SIZE_SOURCE_FIRST_IMAGE, SIZE_SOURCE_MANUAL]" in source
    assert "widthWidget.disabled = firstImage" in source


def _js_function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


def test_size_source_frontend_migrates_legacy_and_disables_the_correct_widgets(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")
    source = (ROOT / "web" / "project_id.js").read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _js_function(source, "findWidget", "setWidgetVisible"),
            _js_function(source, "setWidgetVisible", "hidePersistentWidget"),
            _js_function(source, "hidePersistentWidget", "settingValue"),
            _js_function(source, "attachRefresh", "transientProductionWidgets"),
            _js_function(
                source,
                "configureResolutionPresetWidgets",
                "isOneShotReviewAction",
            ),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const RESOLUTION_PRESET_WIDGET = "preset";
const CUSTOM_MP_WIDGET = "custom_mp";
const CUSTOM_RESOLUTION_PRESET = "Custom";
const LEGACY_ASPECT_WIDGET = "aspect";
const SIZE_SOURCE_WIDGET = "size_source";
const WIDTH_WIDGET = "width";
const HEIGHT_WIDGET = "height";
const SIZE_SOURCE_FIRST_IMAGE = "First Image";
const SIZE_SOURCE_MANUAL = "Manual";
const SIZE_SOURCE_LEGACY = "Legacy Aspect";
const RESOLUTION_PRESET_DRAFT = "Draft — 0.30 MP";
const RESOLUTION_PRESET_BALANCED = "Balanced — 0.60 MP";
const RESOLUTION_PRESET_NATIVE = "Native 768";
const CANVAS_MULTIPLE = 32;
const NATIVE_SHORT_EDGE = 768;
const NATIVE_LONG_EDGE_CAP = 1344;
const imageSource = {{ mode: 0, imgs: [{{ naturalWidth: 1200, naturalHeight: 800 }}] }};
const app = {{ graph: {{ links: {{ 1: {{ origin_id: 9 }} }}, getNodeById: () => imageSource }} }};
{functions}
function widget(name, value) {{
    return {{ name, value, type: "combo", options: {{}}, computeSize: () => [120, 20] }};
}}
function makeNode(aspect, sizeSource, width, height, linked = false) {{
    return {{
        comfyClass: V38_NODE_CLASS,
        widgets: [
            widget(LEGACY_ASPECT_WIDGET, aspect),
            widget(RESOLUTION_PRESET_WIDGET, RESOLUTION_PRESET_BALANCED),
            widget(CUSTOM_MP_WIDGET, 0.3),
            widget(SIZE_SOURCE_WIDGET, sizeSource),
            widget(WIDTH_WIDGET, width),
            widget(HEIGHT_WIDGET, height),
        ],
        inputs: [{{ name: "first_frame", link: linked ? 1 : null }}],
        configure(info) {{
            info.widgets_values.forEach((value, index) => {{
                this.widgets[index].value = value;
            }});
        }},
        setDirtyCanvas() {{}},
    }};
}}
const legacy = makeNode("Landscape 16:9", SIZE_SOURCE_LEGACY, 736, 416);
configureResolutionPresetWidgets(legacy);
const first = makeNode("Auto from First Image", SIZE_SOURCE_FIRST_IMAGE, 736, 416, true);
configureResolutionPresetWidgets(first);
const manual = makeNode("Auto from First Image", SIZE_SOURCE_MANUAL, 704, 1024);
configureResolutionPresetWidgets(manual);
const oldLoad = makeNode("Auto from First Image", SIZE_SOURCE_LEGACY, 736, 416);
configureResolutionPresetWidgets(oldLoad);
oldLoad.configure({{ widgets_values: [
    "Landscape 16:9",
    RESOLUTION_PRESET_BALANCED,
    0.3,
] }});
const savedManual = makeNode("Auto from First Image", SIZE_SOURCE_LEGACY, 736, 416);
configureResolutionPresetWidgets(savedManual);
savedManual.configure({{ widgets_values: [
    "Auto from First Image",
    RESOLUTION_PRESET_BALANCED,
    0.3,
    SIZE_SOURCE_MANUAL,
    704,
    1024,
] }});
function observed(node) {{
    return {{
        source: findWidget(node, SIZE_SOURCE_WIDGET).value,
        sourceOptions: findWidget(node, SIZE_SOURCE_WIDGET).options.values,
        width: findWidget(node, WIDTH_WIDGET).value,
        height: findWidget(node, HEIGHT_WIDGET).value,
        widthDisabled: findWidget(node, WIDTH_WIDGET).disabled,
        presetDisabled: findWidget(node, RESOLUTION_PRESET_WIDGET).disabled,
        aspectHidden: findWidget(node, LEGACY_ASPECT_WIDGET).hidden,
    }};
}}
console.log(JSON.stringify({{
    legacy: observed(legacy),
    first: observed(first),
    manual: observed(manual),
    oldLoad: observed(oldLoad),
    savedManual: observed(savedManual),
}}));
"""
    script_path = tmp_path / "resolution_ux.mjs"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    observed = json.loads(completed.stdout)
    assert observed["legacy"] == {
        "source": "Manual",
        "sourceOptions": ["First Image", "Manual"],
        "width": 1024,
        "height": 576,
        "widthDisabled": False,
        "presetDisabled": True,
        "aspectHidden": True,
    }
    assert observed["first"] == {
        "source": "First Image",
        "sourceOptions": ["First Image", "Manual"],
        "width": 960,
        "height": 640,
        "widthDisabled": True,
        "presetDisabled": False,
        "aspectHidden": True,
    }
    assert observed["manual"] == {
        "source": "Manual",
        "sourceOptions": ["First Image", "Manual"],
        "width": 704,
        "height": 1024,
        "widthDisabled": False,
        "presetDisabled": True,
        "aspectHidden": True,
    }
    assert observed["oldLoad"] == observed["legacy"]
    assert observed["savedManual"] == observed["manual"]
