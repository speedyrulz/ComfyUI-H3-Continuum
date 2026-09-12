from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)
from ComfyUI_H3_Continuum_Join.v3.nodes import (
    _format_review_status,
    _partial_review_warning,
)
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    EXECUTION_MODE_FULL_RUN,
    EXECUTION_MODE_REVIEW_CONTINUE,
    EXECUTION_MODE_REVIEW_FINISH,
    EXECUTION_MODE_REVIEW_REGENERATE,
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_FINISH_REMAINING,
    REVIEW_ACTION_REGENERATE_CURRENT,
    REVISION_STATUS_REVIEW_READY,
    RUN_STORAGE_OFF,
    RUN_STORAGE_SAVE_AUTO_RESUME,
    ReviewControlError,
    resolve_review_execution,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID_JS = ROOT / "web" / "project_id.js"
README = ROOT / "README.md"
README_JA = ROOT / "README_JA.md"


def _resolved(*, mode, action, storage=RUN_STORAGE_SAVE_AUTO_RESUME, manual=0):
    return resolve_review_execution(
        generation_mode=mode,
        review_action=action,
        configured_chunks=4,
        validated_prefix_count=1,
        terminal_merge_enabled=False,
        terminal_pair_start=None,
        manual_regenerate_from=manual,
        run_storage_mode=storage,
        latest_review_unit={"start": 1, "end": 1, "physical_group": 1},
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_effective_nonce=2,
        latest_branch_regenerate_from=1,
    )


def test_e0_e6_backend_queue_intents_keep_phase_b_contract():
    full = _resolved(
        mode=GENERATION_MODE_FULL_RUN,
        action=REVIEW_ACTION_REGENERATE_CURRENT,
        storage=RUN_STORAGE_OFF,
    )
    assert full.execution_mode == EXECUTION_MODE_FULL_RUN
    assert full.max_new_physical_groups is None
    assert full.smart_regenerate is False

    continued = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_CONTINUE,
    )
    assert continued.execution_mode == EXECUTION_MODE_REVIEW_CONTINUE
    assert continued.max_new_physical_groups == 1

    regenerated = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_REGENERATE_CURRENT,
    )
    assert regenerated.execution_mode == EXECUTION_MODE_REVIEW_REGENERATE
    assert regenerated.effective_regenerate_from == 1
    assert regenerated.max_new_physical_groups == 1
    assert regenerated.requested_effective_nonce == 3

    finished = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_FINISH_REMAINING,
    )
    assert finished.execution_mode == EXECUTION_MODE_REVIEW_FINISH
    assert finished.max_new_physical_groups is None

    with pytest.raises(ReviewControlError, match="requires Run Storage"):
        _resolved(
            mode=GENERATION_MODE_REVIEW,
            action=REVIEW_ACTION_CONTINUE,
            storage=RUN_STORAGE_OFF,
        )
    with pytest.raises(ReviewControlError, match="cannot be combined"):
        _resolved(
            mode=GENERATION_MODE_REVIEW,
            action=REVIEW_ACTION_REGENERATE_CURRENT,
            manual=2,
        )

    stale_finish = _resolved(
        mode=GENERATION_MODE_FULL_RUN,
        action=REVIEW_ACTION_FINISH_REMAINING,
        storage=RUN_STORAGE_OFF,
    )
    assert stale_finish.execution_mode == EXECUTION_MODE_FULL_RUN
    assert stale_finish.max_new_physical_groups is None
    assert stale_finish.finish_remaining is False


def test_v38_schema_appends_review_widgets_without_changing_v37():
    v37 = H3ContinuumSamplerV37.INPUT_TYPES()["required"]
    v38 = H3ContinuumSamplerV38.INPUT_TYPES()["required"]
    assert "generation_mode" not in v37
    assert "review_action" not in v37
    # RefMod (`refmod_*`) controls are appended after the review/size widgets.
    assert [name for name in v38 if not name.startswith("refmod_")][-8:] == [
        "generation_mode",
        "review_action",
        "take_group",
        "take_revision_id",
        "take_action",
        "size_source",
        "width",
        "height",
    ]
    assert v38["generation_mode"][1]["default"] == GENERATION_MODE_FULL_RUN
    assert v38["review_action"][1]["default"] == REVIEW_ACTION_CONTINUE
    assert v38["take_action"][1]["default"] == "Automatic"


def test_v38_facade_forwards_only_public_review_intent(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    result = H3ContinuumSamplerV38().run(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=REVIEW_ACTION_FINISH_REMAINING,
    )
    assert result == "ok"
    assert captured["generation_mode"] == GENERATION_MODE_REVIEW
    assert captured["review_action"] == REVIEW_ACTION_FINISH_REMAINING
    assert "max_new_physical_groups" not in captured


def _storage(*, execution, chunks, total, unit=None, reused=0, generated=0):
    manifest = {"chunks": [{} for _ in range(chunks)]}
    if unit is not None:
        manifest["review_unit"] = unit
    return SimpleNamespace(
        review_generation_mode=GENERATION_MODE_REVIEW,
        review_execution=execution,
        manifest=manifest,
        contract={"chunk_count": total},
        reused_count=reused,
        generated_count=generated,
    )


def test_review_status_covers_ready_terminal_regenerate_and_finish():
    ready = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=False,
            partial_review=True,
        ),
        chunks=2,
        total=6,
        unit={"start": 2, "end": 2, "physical_group": 2},
    )
    ready_status = _format_review_status(ready)
    assert "Chunk 2 / 6 ready" in ready_status
    assert "Completed: Chunks 1-2" in ready_status
    assert "Queue again = Accept + Continue" in ready_status

    terminal = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=False,
            partial_review=False,
        ),
        chunks=3,
        total=3,
        unit={"start": 2, "end": 3, "physical_group": 2},
    )
    terminal_status = _format_review_status(terminal)
    assert "Chunks 2-3 / 3 ready" in terminal_status
    assert "Terminal Merge: 1 physical review unit" in terminal_status
    assert terminal_status.endswith("Sequence complete")

    regenerated = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=True,
            requested_effective_nonce=3,
            partial_review=True,
        ),
        chunks=3,
        total=6,
        unit={"start": 3, "end": 3, "physical_group": 3},
    )
    regenerated_status = _format_review_status(regenerated, detailed=True)
    assert "Smart Regenerate" in regenerated_status
    assert "Chunk 3 regenerated" in regenerated_status
    assert "Preserved: Chunks 1-2" in regenerated_status
    assert "Variation: 3" in regenerated_status

    finished = _storage(
        execution=SimpleNamespace(
            finish_remaining=True,
            smart_regenerate=False,
            partial_review=False,
        ),
        chunks=6,
        total=6,
        reused=3,
        generated=3,
    )
    finished_status = _format_review_status(finished)
    assert finished_status == (
        "Review completed\n3 reused; 3 generated; 6 total\nSequence complete"
    )


def test_partial_review_second_pass_warning_is_explicit():
    partial = SimpleNamespace(
        review_execution=SimpleNamespace(partial_review=True)
    )
    assert "Second Pass refinement" in _partial_review_warning(
        partial,
        capture_refine_context=True,
    )
    assert _partial_review_warning(
        partial,
        capture_refine_context=False,
    ) == ""


def _function_source(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


def test_f0_f7_frontend_review_lifecycle(review_queue_results):
    # Acceptance, not a manual helper call, must consume one-shot intent.
    for case in (
        "three review queues advance without any executed events",
        "queue rejection preserves one-shot retry for resubmission",
        "retry consumed after acceptance without legacy preparation hook",
        "mode policy and transient UI serialization remain stable",
    ):
        assert review_queue_results[case]["pass"]

def test_frontend_uses_scoped_acceptance_adapter_without_app_queue_override():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    assert "actionWidget.afterQueued = function" in source
    assert "prepareReviewQueueIntent(node, apiNode.inputs);" in source
    assert "normalizeReviewActionOnLoad(node);" in source
    assert "app.queuePrompt =" not in source
    assert "api.__h3ContinuumReviewQueueAdapter" in source


def test_execution_success_reloads_v38_history_without_auto_queue():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    refresh = _function_source(
        source,
        "refreshV38TakeHistoryAfterExecution",
        "attachTakeHistoryReload",
    )
    setup = source[source.index("setup() {") : source.index("nodeCreated(node)")]
    for event_name in (
        "execution_success",
        "execution_error",
        "execution_interrupted",
    ):
        assert f'"{event_name}"' in setup
    assert "refreshV38TakeHistoryAfterExecution" in setup
    assert "node.comfyClass !== V38_NODE_CLASS" in refresh
    assert "await loadTakeHistory(node," in refresh
    assert "reviewPrompts.get(promptId)" in refresh
    assert "event?.detail?.prompt_id" in refresh
    assert "api.queuePrompt(" not in refresh


def test_review_requires_fixed_control_after_generate(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "baseSeedControlWidget", "setReviewSeedControlFixed"),
            _function_source(source, "requireFixedSeedForReview", "readySummary"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const GENERATION_MODE_WIDGET = "generation_mode";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
{functions}
function node(mode, control, comfyClass = V38_NODE_CLASS) {{
    const linkedControl = {{ name: "control_after_generate", value: control }};
    return {{
        comfyClass,
        widgets: [
            {{ name: GENERATION_MODE_WIDGET, value: mode }},
            {{ name: "base_seed", value: 123, linkedWidgets: [linkedControl] }},
            {{ name: "control_after_generate", value: control === "fixed" ? "randomize" : "fixed" }},
        ],
    }};
}}
function result(value) {{
    try {{ requireFixedSeedForReview(value); return "allowed"; }}
    catch (error) {{ return String(error.message); }}
}}
console.log(JSON.stringify({{
    fullRandom: result(node("Full Run", "randomize")),
    reviewFixed: result(node(GENERATION_MODE_REVIEW, "fixed")),
    reviewRandom: result(node(GENERATION_MODE_REVIEW, "randomize")),
    legacyReviewRandom: result(node(GENERATION_MODE_REVIEW, "randomize", "H3ContinuumSamplerV37")),
}}));
"""
    script_path = tmp_path / "v38-review-fixed-seed.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["fullRandom"] == "allowed"
    assert observed["reviewFixed"] == "allowed"
    assert observed["legacyReviewRandom"] == "allowed"
    assert "Control After Generate = fixed" in observed["reviewRandom"]


def test_review_mode_switch_fixes_the_linked_seed_control(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setWidgetVisible", "hidePersistentWidget"),
            _function_source(source, "setExistingWidgetValue", "facadeProductionWidgets"),
            _function_source(source, "baseSeedControlWidget", "setReviewSeedControlFixed"),
            _function_source(source, "setReviewSeedControlFixed", "requireFixedSeedForReview"),
            _function_source(source, "configureReviewControls", "configureAssembler"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const GENERATION_MODE_WIDGET = "generation_mode";
const REVIEW_ACTION_WIDGET = "review_action";
const RUN_STORAGE_WIDGET = "run_storage";
const TAKE_ACTION_WIDGET = "take_action";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
{functions}
const linkedControl = {{ name: "control_after_generate", value: "randomize" }};
const decoyControl = {{ name: "control_after_generate", value: "fixed" }};
const generation = {{ name: GENERATION_MODE_WIDGET, value: "Full Run" }};
const storage = {{ name: RUN_STORAGE_WIDGET, value: "Off" }};
const node = {{
    comfyClass: V38_NODE_CLASS,
    widgets: [
        generation,
        {{ name: REVIEW_ACTION_WIDGET, value: "Continue / Next" }},
        storage,
        {{ name: TAKE_ACTION_WIDGET, value: "Automatic" }},
        {{ name: "base_seed", value: 123, linkedWidgets: [linkedControl] }},
        decoyControl,
    ],
    setDirtyCanvas() {{}},
}};
configureReviewControls(node);
generation.value = GENERATION_MODE_REVIEW;
generation.callback(GENERATION_MODE_REVIEW);
console.log(JSON.stringify({{
    linked: linkedControl.value,
    decoy: decoyControl.value,
    storage: storage.value,
}}));
"""
    script_path = tmp_path / "v38-review-linked-seed-control.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "linked": "fixed",
        "decoy": "fixed",
        "storage": "Save + Auto Resume",
    }


def test_readmes_document_exact_two_by_five_review_workflow():
    for path in (README, README_JA):
        text = path.read_text(encoding="utf-8")
        for label in (
            "`Chunks`",
            "`Seconds per Chunk`",
            "`Total Length`",
            "`Run`",
            "`Review Each Chunk`",
            "`Progress`",
            "`On — Resume and Takes available`",
            "`Ready to Queue`",
            "`Control After Generate`",
            "`fixed`",
            "`Use it and continue`",
            "`Try this chunk again`",
            "`Use it and finish the rest`",
            "`Back to Settings`",
            "`Return to Review`",
            "`Queue`",
        ):
            assert label in text
        assert "5" in text
        assert "10" in text


def test_n1_production_shortcuts_are_transient_and_map_to_existing_intent(review_queue_results):
    for case in (
        "mode policy and transient UI serialization remain stable",
        "explicit restart consumed and Chunk 1 Continue restored",
        "the caller graph and workflow are not mutated by Queue adapter",
        "history browsing never changes serialized selection, explicit apply does",
    ):
        assert review_queue_results[case]["pass"]

def test_n2c_render_history_take_ux_is_complete_and_backend_derived(review_queue_results):
    for case in (
        "history catalog preserves lineage eligibility and compact summary",
        "history browsing never changes serialized selection, explicit apply does",
        "late response for an old Run cannot replace the current history",
        "fresh browser state can open persisted Review without prior events",
    ):
        assert review_queue_results[case]["pass"]

def test_n2c_history_refresh_hooks_and_multiline_surface_are_present():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    assert '"Render History / Takes"' in source
    assert '{ multiline: true }' in source
    assert "await loadTakeHistory(node," in source
    assert "attachTakeHistoryReload(node);" in source
    assert "node.__h3ContinuumTakeInitialLoad" in source
    assert 'const TAKE_TOGGLE_WIDGET = "Render History";' in source
    assert "app.queuePrompt =" not in source
    assert "api.__h3ContinuumReviewQueueAdapter" in source
