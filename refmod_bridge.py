"""Optional MiniMax H3 RefMod integration for the Continuum Sampler.

`ComfyUI-MiniMaxH3Mod` ("RefMod") saves compressed H3 image/video/audio
references and exposes them as an ``H3_REF_MODS`` bundle from its
``Load H3 RefMods`` / ``Load H3 RefMod Axis`` / ``Create H3 RefMod`` nodes.
Its ``Apply H3 RefMod`` node turns that bundle into native reference blocks
and appends them to a single conditioning.  Continuum builds one conditioning
per chunk internally, so the Sampler cannot accept an already-applied
conditioning; instead it accepts the bundle directly, exposes the same
adjustable Apply settings, and injects the resolved blocks into every chunk's
guider conditioning through the public ``OUTER_SAMPLE`` model wrapper.

Design constraints:

- The RefMod pack is optional.  Nothing here imports it at module load; the
  installed pack is discovered lazily through ``sys.modules`` only when a
  bundle is actually connected.  When its ``_ref_blocks`` helper is present
  it is used so the Sampler matches ``Apply H3 RefMod`` exactly; otherwise a
  duck-typed fallback with the same semantics builds the blocks from the mod
  objects' own ``ref_block`` method.
- Injection is model-scoped (a cloned MODEL with one keyed wrapper), so it
  composes with SageAttention, Sol-Attn, Spectrum, and the pack's own
  ``H3 RefMod Step Curve`` wrapper, and it leaves Continuum's chunk
  conditioning, continuation transport, Audio, Seed, and SIGMAS untouched.
- A disconnected bundle is a complete no-op: the MODEL is passed through
  unchanged and no wrapper is added.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Callable

REFMODS_TYPE = "H3_REF_MODS"
BRIDGE_WRAPPER_KEY = "h3_continuum.refmod_bridge.v1"
OUTER_SAMPLE_FALLBACK = "outer_sample"

# Mirrors ``core.CURVE_DIRECTIONS`` / ``core.CURVE_SHAPES`` of the RefMod pack
# so the Sampler schema is stable even when the pack is not installed.
CURVE_DIRECTIONS = (
    "constant",
    "concept_at_start",
    "concept_at_middle",
    "concept_at_end",
    "concept_at_ends",
)
CURVE_SHAPES = (
    "linear",
    "ease",
    "sigmoid",
    "tanh",
    "quadratic",
    "cubic",
    "exponential",
    "stair",
    "elastic",
    "bump",
    "dip",
)
SCRAMBLE_MODES = ("shuffle", "subset", "legacy_subset")

DEFAULT_RETENTION = 1.0
DEFAULT_CURVE_DIRECTION = "constant"
DEFAULT_CURVE_SHAPE = "linear"
DEFAULT_CURVE_VALUE = 1.0
DEFAULT_SCRAMBLE_SEED = -1
DEFAULT_SCRAMBLE_MODE = "shuffle"
DEFAULT_SCRAMBLE_KEEP = 1
DEFAULT_MAX_TOKENS = 0

# Loose upper bound used only for the diagnostic note when the pack helper
# is unavailable; the pack itself decides whether a budget is a hard error.
_REFMOD_PACK_MARKERS = ("_ref_blocks", "MiniMaxH3RefModApply", "MiniMaxH3RefModsLoader")


class RefModBridgeError(ValueError):
    """Raised when a connected bundle cannot be turned into reference blocks."""


def resolve_refmod_pack() -> Any | None:
    """Return the loaded RefMod ``nodes`` module, or ``None`` when absent.

    ComfyUI registers custom-node packages under their folder name, which the
    user may have renamed, so the lookup is by capability rather than by
    module name.
    """
    for name, module in list(sys.modules.items()):
        if module is None or not name.endswith("nodes"):
            continue
        try:
            if all(hasattr(module, marker) for marker in _REFMOD_PACK_MARKERS):
                return module
        except Exception:
            continue
    return None


def _clamp01(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def normalize_settings(
    *,
    retention: Any = DEFAULT_RETENTION,
    curve_direction: Any = DEFAULT_CURVE_DIRECTION,
    curve_shape: Any = DEFAULT_CURVE_SHAPE,
    curve_value: Any = DEFAULT_CURVE_VALUE,
    scramble_seed: Any = DEFAULT_SCRAMBLE_SEED,
    scramble_mode: Any = DEFAULT_SCRAMBLE_MODE,
    scramble_keep: Any = DEFAULT_SCRAMBLE_KEEP,
    max_total_tokens: Any = DEFAULT_MAX_TOKENS,
    override: Any = False,
) -> dict[str, Any]:
    """Coerce Sampler widget values to the exact types the Apply path expects.

    Unknown combo values fall back to their defaults instead of stopping the
    run; the returned ``notes`` list records every substitution so the report
    can show it.
    """
    notes: list[str] = []
    direction = str(curve_direction or DEFAULT_CURVE_DIRECTION)
    if direction not in CURVE_DIRECTIONS:
        notes.append(f"unknown curve direction {direction!r}; using {DEFAULT_CURVE_DIRECTION}")
        direction = DEFAULT_CURVE_DIRECTION
    shape = str(curve_shape or DEFAULT_CURVE_SHAPE)
    if shape not in CURVE_SHAPES:
        notes.append(f"unknown curve shape {shape!r}; using {DEFAULT_CURVE_SHAPE}")
        shape = DEFAULT_CURVE_SHAPE
    mode = str(scramble_mode or DEFAULT_SCRAMBLE_MODE)
    if mode not in SCRAMBLE_MODES:
        notes.append(f"unknown scramble mode {mode!r}; using {DEFAULT_SCRAMBLE_MODE}")
        mode = DEFAULT_SCRAMBLE_MODE
    try:
        seed = int(scramble_seed)
    except (TypeError, ValueError):
        seed = DEFAULT_SCRAMBLE_SEED
    try:
        keep = max(1, int(scramble_keep))
    except (TypeError, ValueError):
        keep = DEFAULT_SCRAMBLE_KEEP
    try:
        budget = max(0, int(max_total_tokens))
    except (TypeError, ValueError):
        budget = DEFAULT_MAX_TOKENS
    return {
        "retention": _clamp01(retention, DEFAULT_RETENTION),
        "curve_direction": direction,
        "curve_shape": shape,
        "curve_value": _clamp01(curve_value, DEFAULT_CURVE_VALUE),
        "scramble_seed": seed,
        "scramble_mode": mode,
        "scramble_keep": keep,
        "max_total_tokens": budget,
        "override": bool(override),
        "notes": notes,
    }


def bundle_items(mods: Any) -> list[tuple[Any, float]]:
    """Validate an ``H3_REF_MODS`` payload into ``[(mod, strength), ...]``."""
    if mods is None:
        return []
    if not isinstance(mods, (list, tuple)):
        raise RefModBridgeError(
            "refmods must be an H3_REF_MODS bundle from Load H3 RefMods / Load H3 RefMod Axis"
        )
    items: list[tuple[Any, float]] = []
    for entry in mods:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise RefModBridgeError("refmods bundle entries must be (mod, strength) pairs")
        mod, strength = entry
        if not callable(getattr(mod, "ref_block", None)):
            raise RefModBridgeError(
                f"refmods entry {getattr(mod, 'name', mod)!r} is not a RefMod object"
            )
        items.append((mod, _clamp01(strength, 1.0)))
    return items


def _saved_curve(config: Any, key: str) -> tuple[str, str, float] | None:
    entry = (config or {}).get(key) if isinstance(config, dict) else None
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        return None
    direction, shape, value = entry
    if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
        return None
    try:
        return (str(direction), str(shape), float(value))
    except (TypeError, ValueError):
        return None


def resolve_override(items: list[tuple[Any, float]], settings: dict[str, Any]) -> dict[str, Any]:
    """Apply the ``override`` toggle exactly like ``Apply H3 RefMod``.

    The first mod carrying a saved ``curve`` config supplies the curve and,
    when present, the retention.  Without one the manual settings stay in
    force and a note is recorded.
    """
    resolved = dict(settings)
    resolved["curve_source"] = "sampler widgets"
    resolved["retention_source"] = "sampler widget"
    if not settings.get("override"):
        return resolved
    for mod, _strength in items:
        config = getattr(mod, "config", None)
        saved = _saved_curve(config, "curve")
        if saved is None:
            continue
        resolved["curve_direction"], resolved["curve_shape"], resolved["curve_value"] = saved
        resolved["curve_value"] = _clamp01(resolved["curve_value"], DEFAULT_CURVE_VALUE)
        resolved["curve_source"] = f"saved config of '{getattr(mod, 'name', '?')}'"
        retention = config.get("retention") if isinstance(config, dict) else None
        if isinstance(retention, (int, float)) and not isinstance(retention, bool):
            resolved["retention"] = _clamp01(retention, DEFAULT_RETENTION)
            resolved["retention_source"] = resolved["curve_source"]
        return resolved
    resolved.setdefault("notes", [])
    resolved["notes"] = list(resolved["notes"]) + [
        "override is on but no mod in the bundle carries a saved curve config; using the sampler widgets"
    ]
    return resolved


def _curve_spec(settings: dict[str, Any]) -> tuple[str, str, float] | None:
    flat = (
        settings["curve_direction"] == "constant"
        and settings["curve_shape"] == "linear"
        and float(settings["curve_value"]) >= 1.0
    )
    if flat:
        return None
    return (settings["curve_direction"], settings["curve_shape"], float(settings["curve_value"]))


def _fallback_blocks(items: list[tuple[Any, float]], settings: dict[str, Any]) -> list[dict]:
    """Duck-typed port of the pack's ``_ref_blocks`` used when it is unavailable."""
    factor = float(settings["retention"])
    curve = _curve_spec(settings)
    ordered = list(items)
    seed = int(settings["scramble_seed"])
    if seed >= 0 and len(ordered) > 1:
        rng = random.Random(seed)
        rng.shuffle(ordered)
        mode = settings["scramble_mode"]
        if mode == "legacy_subset":
            ordered = ordered[: rng.randint(max(1, len(ordered) // 2), len(ordered))]
        elif mode == "subset":
            ordered = ordered[: max(1, int(settings["scramble_keep"]))]
    budget = int(settings["max_total_tokens"])
    if budget > 0:
        total = sum(int(getattr(mod, "token_count", 0) or 0) for mod, _ in ordered)
        if total > budget:
            raise RefModBridgeError(
                f"refmods bundle uses {total} reference tokens, above the {budget} token budget"
            )
    blocks: list[dict] = []
    for mod, strength in ordered:
        effective = min(1.0, max(0.0, strength * factor))
        used_curve = curve
        if int(getattr(mod, "latent_t", 1) or 1) <= 1 and curve is not None and curve[0] != "constant":
            # Directions run across a mod's own frames; a one-frame mod only
            # honours curve_value as a plain strength cap (Apply parity).
            effective = min(effective, max(0.0, min(1.0, float(curve[2]))))
            used_curve = None
        block = mod.ref_block(effective, curve=used_curve)
        if block is None:
            continue
        block = dict(block)
        block["refmod"] = True
        blocks.append(block)
    return blocks


def build_refmod_blocks(
    mods: Any,
    settings: dict[str, Any],
    *,
    pack: Any | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Resolve a bundle into native ref blocks plus a diagnostics summary.

    ``settings`` is the output of :func:`normalize_settings`.  Returns
    ``(blocks, summary)`` where ``summary`` carries the resolved settings,
    the helper used, and the per-mod effective strengths for the report.
    """
    items = bundle_items(mods)
    resolved = resolve_override(items, settings)
    summary: dict[str, Any] = {
        "mod_count": len(items),
        "mods": [
            (str(getattr(mod, "name", "?")), float(strength), int(getattr(mod, "token_count", 0) or 0))
            for mod, strength in items
        ],
        "settings": resolved,
        "helper": "none",
        "notes": list(resolved.get("notes", [])),
    }
    if not items:
        return [], summary
    if pack is None:
        pack = resolve_refmod_pack()
    helper: Callable[..., list[dict]] | None = getattr(pack, "_ref_blocks", None) if pack else None
    if callable(helper):
        summary["helper"] = f"{pack.__name__}._ref_blocks"
        blocks = helper(
            items,
            float(resolved["retention"]),
            _curve_spec(resolved),
            seed=int(resolved["scramble_seed"]),
            scramble_mode=str(resolved["scramble_mode"]),
            scramble_keep=int(resolved["scramble_keep"]),
            max_total_tokens=int(resolved["max_total_tokens"]),
        )
        blocks = [dict(block) for block in blocks]
    else:
        summary["helper"] = "continuum fallback"
        blocks = _fallback_blocks(items, resolved)
    summary["block_count"] = len(blocks)
    summary["token_count"] = sum(_block_tokens(block) for block in blocks)
    summary["block_stats"] = [_block_stats(block) for block in blocks]
    return blocks, summary


def _tensor_stats(tensor: Any) -> dict[str, Any] | None:
    """Small CPU summary of a ref latent used to spot NaN/Inf or wild scaling."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return None
    if not torch.is_tensor(tensor):
        return None
    try:
        value = tensor.detach().to("cpu", dtype=torch.float32)
        finite = bool(torch.isfinite(value).all().item())
        clean = value if finite else value[torch.isfinite(value)]
        return {
            "shape": tuple(int(v) for v in tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "finite": finite,
            "absmax": float(clean.abs().max().item()) if clean.numel() else 0.0,
            "mean": float(clean.mean().item()) if clean.numel() else 0.0,
            "std": float(clean.std(unbiased=False).item()) if clean.numel() else 0.0,
        }
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def _block_stats(block: dict) -> dict[str, Any]:
    return {
        "kind": block.get("kind"),
        "latent_t": int(block.get("latent_t", 1) or 1),
        "latent_h": int(block.get("latent_h", 0) or 0),
        "latent_w": int(block.get("latent_w", 0) or 0),
        "ref_audio_t": int(block.get("ref_audio_t", 0) or 0),
        "latent": _tensor_stats(block.get("latent")),
        "audio_latent": _tensor_stats(block.get("audio_latent")),
    }


def _block_tokens(block: dict) -> int:
    kind = block.get("kind")
    if kind == "audio":
        return 2 * int(block.get("ref_audio_t", 0) or 0)
    per_frame = (int(block.get("latent_h", 0) or 0) // 2) * (int(block.get("latent_w", 0) or 0) // 2)
    return per_frame * int(block.get("latent_t", 1) or 1)


def _outer_sample_key() -> str:
    try:
        from comfy.patcher_extension import WrappersMP

        return str(WrappersMP.OUTER_SAMPLE)
    except Exception:
        return OUTER_SAMPLE_FALLBACK


def attach_refmod_blocks(model: Any, blocks: list[dict]) -> Any:
    """Return a cloned MODEL whose sampling injects ``blocks`` into every chunk.

    Uses the public ``OUTER_SAMPLE`` wrapper so it runs once per
    ``CFGGuider.sample`` call, which is exactly once per Continuum physical
    sampling group.  The guider's conditioning is restored after the call,
    also on exception or cancellation, so nothing leaks across chunks.
    """
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise RefModBridgeError("model does not expose the ComfyUI ModelPatcher wrapper API")
    clone = model.clone()
    wrapper_group = _outer_sample_key()
    remover = getattr(clone, "remove_wrappers_with_key", None)
    if callable(remover):
        remover(wrapper_group, BRIDGE_WRAPPER_KEY)
    if not blocks:
        return clone
    refs = tuple(dict(block) for block in blocks)

    def inject(executor, *args, **kwargs):
        guider = executor.class_obj
        original = guider.conds
        guider.conds = {
            key: [
                dict(entry, minimax_refs=list(entry.get("minimax_refs") or []) + [dict(ref) for ref in refs])
                for entry in entries
            ]
            for key, entries in original.items()
        }
        try:
            return executor(*args, **kwargs)
        finally:
            guider.conds = original

    clone.add_wrapper_with_key(wrapper_group, BRIDGE_WRAPPER_KEY, inject)
    return clone


def format_refmod_status(summary: dict[str, Any]) -> str:
    """Report lines describing what was injected (or why nothing was)."""
    lines = ["RefMod Bridge"]
    count = int(summary.get("mod_count", 0))
    if count == 0:
        lines.append("Bundle connected but empty (all slots (none) or strength 0); MODEL passed through unchanged.")
    else:
        settings = summary.get("settings", {})
        mods = ", ".join(
            f"{name}@{strength:.2f}" + (f" ({tokens} tok)" if tokens else "")
            for name, strength, tokens in summary.get("mods", [])
        )
        lines.append(
            f"{summary.get('block_count', 0)} reference block(s) from {count} mod(s) injected into every chunk: {mods}."
        )
        curve = (
            f"{settings.get('curve_direction')} / {settings.get('curve_shape')} @ "
            f"{float(settings.get('curve_value', 1.0)):.2f}"
        )
        seed = int(settings.get("scramble_seed", -1))
        scramble = "off" if seed < 0 or count < 2 else (
            f"seed {seed}, {settings.get('scramble_mode')}"
            + (f" keep {settings.get('scramble_keep')}" if settings.get("scramble_mode") == "subset" else "")
        )
        lines.append(
            f"retention={float(settings.get('retention', 1.0)):.2f} ({settings.get('retention_source', 'sampler widget')}); "
            f"curve={curve} ({settings.get('curve_source', 'sampler widgets')}); scramble={scramble}; "
            f"tokens={summary.get('token_count', 0)}; helper={summary.get('helper', 'none')}."
        )
    non_finite = []
    for index, stats in enumerate(summary.get("block_stats", []), start=1):
        parts = [
            f"kind={stats.get('kind')}",
            f"t={stats.get('latent_t')}",
            f"hw={stats.get('latent_h')}x{stats.get('latent_w')}",
        ]
        if stats.get("ref_audio_t"):
            parts.append(f"audio_t={stats['ref_audio_t']}")
        for label in ("latent", "audio_latent"):
            ts = stats.get(label)
            if not ts:
                continue
            if "error" in ts:
                parts.append(f"{label}={ts['error']}")
                continue
            if ts["finite"] is False:
                non_finite.append(index)
            parts.append(
                f"{label}={ts['shape']} {ts['dtype']}@{ts['device']} "
                f"finite={ts['finite']} absmax={ts['absmax']:.3f} "
                f"mean={ts['mean']:.3f} std={ts['std']:.3f}"
            )
        lines.append(f"Block {index}: " + "; ".join(parts))
    if non_finite:
        lines.append(
            "WARNING: reference block(s) "
            + ", ".join(str(i) for i in non_finite)
            + " contain NaN/Inf; an injected non-finite reference poisons every "
            "chunk (black output). Re-create the mod or set its strength to 0."
        )
    for note in summary.get("notes", []):
        lines.append(f"Note: {note}")
    return "\n".join(lines)
