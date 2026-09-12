"""Public V3.4 Driving Audio sampler and assembler nodes."""

from __future__ import annotations

import torch

from ..branch_provenance import (
    TAKE_ACTION_AUTOMATIC,
    TAKE_ACTION_OPTIONS,
)
from ..constants import (
    CONTINUITY_OPTIONS,
    DIAGNOSTICS_FULL,
    DIAGNOSTICS_OFF,
    FPS,
    normalize_diagnostics_mode,
)
from ..driving_audio import prepare_driving_audio_source
from ..guide_timeline import (
    GUIDE_TYPE,
    make_still_image_guide,
    prepare_still_image_guide_source,
)
from ..hardening import diagnostics_is_full
from ..reference_video import (
    REFERENCE_VIDEO_SIZE_EFFICIENT,
    REFERENCE_VIDEO_SIZE_OPTIONS,
)
from ..reference_audio import (
    H3ContinuumReferenceAudios,
    REFERENCE_AUDIOS_TYPE,
)
from ..refmod_bridge import (
    CURVE_DIRECTIONS as REFMOD_CURVE_DIRECTIONS,
    CURVE_SHAPES as REFMOD_CURVE_SHAPES,
    DEFAULT_CURVE_DIRECTION as REFMOD_DEFAULT_CURVE_DIRECTION,
    DEFAULT_CURVE_SHAPE as REFMOD_DEFAULT_CURVE_SHAPE,
    DEFAULT_CURVE_VALUE as REFMOD_DEFAULT_CURVE_VALUE,
    DEFAULT_MAX_TOKENS as REFMOD_DEFAULT_MAX_TOKENS,
    DEFAULT_RETENTION as REFMOD_DEFAULT_RETENTION,
    DEFAULT_SCRAMBLE_KEEP as REFMOD_DEFAULT_SCRAMBLE_KEEP,
    DEFAULT_SCRAMBLE_MODE as REFMOD_DEFAULT_SCRAMBLE_MODE,
    DEFAULT_SCRAMBLE_SEED as REFMOD_DEFAULT_SCRAMBLE_SEED,
    REFMODS_TYPE,
    SCRAMBLE_MODES as REFMOD_SCRAMBLE_MODES,
    attach_refmod_blocks,
    build_refmod_blocks,
    format_refmod_status,
    normalize_settings as normalize_refmod_settings,
)
from ..v2.decoder import enforce_total_frames
from .assembly import (
    H3ContinuumAssembleSeamExperimental,
    _chunk_list,
    _singleton,
)
from .assembly_v35 import (
    BUFFER_BACKEND_AUTO,
    BUFFER_BACKEND_OPTIONS,
    assemble_decoded_chunks_v35,
)
from .memory_attribution import (
    format_assembly_projection,
    project_assembly_buffers,
)
from .memory_action_policy import (
    ACTION_DISABLED,
    ACTION_OPTIONS,
    MEMORY_ACTION_POLICY_TYPE,
    format_memory_action_status,
    make_memory_action_policy,
    resolve_memory_action_policy,
)
from .nodes import CATEGORY as CONTINUUM_CATEGORY, H3ContinuumSamplerProduction
from .resolution import (
    H3_ASPECT_AUTO,
    H3_ASPECT_OPTIONS,
    H3_CANVAS_MAX,
    H3_CANVAS_MIN,
    H3_CANVAS_MULTIPLE,
    H3_CUSTOM_MP_DEFAULT,
    H3_CUSTOM_MP_MAX,
    H3_CUSTOM_MP_MIN,
    H3_MANUAL_HEIGHT_DEFAULT,
    H3_MANUAL_WIDTH_DEFAULT,
    H3_PRESET_DRAFT,
    H3_PRESET_OPTIONS,
    H3_SIZE_SOURCE_LEGACY,
    H3_SIZE_SOURCE_OPTIONS,
    resolve_h3_size_source,
)
from .reliability_v38 import append_v38_status, build_v38_diagnostics
from .review_control import (
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_OPTIONS,
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_OPTIONS,
)


_DRIVING_AUDIO_PLAN_KEY = "_h3_continuum_driving_audio_v1"


def _unwrap_single_audio_value(value):
    while isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    return value


def _copy_audio(value):
    value = _unwrap_single_audio_value(value)
    if not isinstance(value, dict):
        return None
    waveform = value.get("waveform")
    sample_rate = value.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        return None
    return {
        "waveform": waveform.detach().to("cpu").contiguous().clone(),
        "sample_rate": int(sample_rate),
    }


def _driving_audio_from_plan(args, kwargs):
    value = kwargs.get("assembly_plan")
    if value is None and len(args) >= 3:
        value = args[2]
    value = _unwrap_single_audio_value(value)
    if not isinstance(value, dict):
        return None
    return _copy_audio(value.get(_DRIVING_AUDIO_PLAN_KEY))


class H3ContinuumSamplerV34(H3ContinuumSamplerProduction):
    """V3.4 sampler with globally encoded, absolute-time Driving Audio."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.4 with optional Driving Audio and persistent Video Reference. "
        "Video Reference uses an IMAGE frame batch and one-chunk reference prefix."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.4",
        "MiniMax H3 Driving Audio",
    ]
    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
        "AUDIO",
    )
    RETURN_NAMES = (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema.get("required", {}))
        required["video_reference_size"] = (
            REFERENCE_VIDEO_SIZE_OPTIONS,
            {
                "default": REFERENCE_VIDEO_SIZE_EFFICIENT,
                "display_name": "Video Guide Size",
                "tooltip": (
                    "Efficient limits Video Guide Frames to about 0.4 MP; Balanced uses "
                    "about 0.6 MP; Match Output uses the output pixel area. Source "
                    "aspect ratio is preserved and smaller sources are not enlarged."
                ),
            },
        )
        optional = dict(schema.get("optional", {}))
        optional.pop("reference_audio_1", None)
        optional.pop("reference_audio_vae", None)
        optional["reference_video_1"] = (
            "IMAGE",
            {
                "display_name": "Video Guide Frames",
                "tooltip": (
                    "Optional video guide. Connect the IMAGE frame batch from a video loader; "
                    "source-video audio is not included. Frames are interpreted at 24 fps "
                    "and applied to every chunk. Non-native frame counts are padded by "
                    "repeating the final frame to the next H3 17k+5 count, up to the "
                    "one-chunk limit."
                ),
            },
        )
        optional["driving_audio"] = (
            "AUDIO",
            {
                "display_name": "Driving Audio",
                "tooltip": (
                    "Optional original audio timeline. It is used as native H3 guide "
                    "conditioning and selected unchanged for final output."
                )
            },
        )
        optional["audio_vae"] = (
            "VAE",
            {
                "display_name": "Driving Audio VAE",
                "tooltip": (
                    "Required only when Driving Audio is connected. Uses the same "
                    "Audio VAE encode path as ComfyUI Core MiniMax H3 Add Guide."
                )
            },
        )
        schema["required"] = required
        schema["optional"] = optional
        return schema

    def run(
        self,
        driving_audio=None,
        audio_vae=None,
        reference_video_1=None,
        video_reference_size=REFERENCE_VIDEO_SIZE_EFFICIENT,
        capture_refine_context=False,
        reference_audio_1=None,
        reference_audio_vae=None,
        **kwargs,
    ):
        from ..reference_video import prepare_reference_video_source
        from ..temporal import align_frame_count_up

        target_frames = round(
            int(kwargs["chunks"]) * float(kwargs["chunk_seconds"]) * FPS
        )
        source = prepare_driving_audio_source(
            driving_audio,
            audio_vae,
            target_frames=target_frames,
            fps=FPS,
        )
        reference_video_source = prepare_reference_video_source(
            reference_video_1,
            target_frames=align_frame_count_up(
                int(round(float(kwargs["chunk_seconds"]) * FPS))
            ),
            output_width=int(kwargs["width"]),
            output_height=int(kwargs["height"]),
            size_mode=str(video_reference_size),
        )
        outputs = super().run(
            reference_audio_1=reference_audio_1,
            reference_audio_vae=reference_audio_vae,
            driving_audio_source=source,
            driving_audio_vae=audio_vae,
            reference_video_source=reference_video_source,
            capture_refine_context=bool(capture_refine_context),
            **kwargs,
        )
        selected_audio = _copy_audio(source.source_audio) if source is not None else None
        if selected_audio is not None and len(outputs) >= 3 and isinstance(outputs[2], dict):
            assembly_plan = dict(outputs[2])
            assembly_plan[_DRIVING_AUDIO_PLAN_KEY] = _copy_audio(selected_audio)
            outputs = (*outputs[:2], assembly_plan, *outputs[3:])
        if bool(capture_refine_context):
            return (*outputs[:4], selected_audio, outputs[4])
        return (*outputs, selected_audio)


class H3ContinuumSamplerV35(H3ContinuumSamplerV34):
    """V3.5 sampler with a physical-group First Pass refine context output."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.5 sampler. It preserves the V3.4 outputs and adds one "
        "physical-group refine context for context-aware Hi-Res Fix or an "
        "external latent Second Pass."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.5",
        "MiniMax H3 context-aware Hi-Res Fix",
    ]
    RETURN_TYPES = (
        *H3ContinuumSamplerV34.RETURN_TYPES,
        "H3_CONTINUUM_REFINE_CONTEXT",
    )
    RETURN_NAMES = (
        *H3ContinuumSamplerV34.RETURN_NAMES,
        "refine_context",
    )
    OUTPUT_IS_LIST = (*H3ContinuumSamplerV34.OUTPUT_IS_LIST, False)

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional["reference_audio_1"] = (
            "AUDIO",
            {
                "display_name": "Reference Audio (Optional)",
                "tooltip": (
                    "Optional standalone audio reference for H3 conditioning. "
                    "It is not the audio track of Video Guide Frames. Unlike Driving "
                    "Audio, it does not replace the generated final audio."
                ),
            },
        )
        optional["reference_audio_vae"] = (
            "VAE",
            {
                "display_name": "Reference Audio VAE (Optional)",
                "tooltip": (
                    "Required only when Reference Audio is connected. It encodes the "
                    "reference for H3 conditioning; generated audio remains the output."
                ),
            },
        )
        schema["optional"] = optional
        return schema

    def run(self, *args, **kwargs):
        kwargs.pop("capture_refine_context", None)
        kwargs.pop("memory_attribution", None)
        kwargs.pop("prompt_conditioning_cache", None)
        return super().run(
            *args,
            capture_refine_context=True,
            memory_attribution=True,
            prompt_conditioning_cache=True,
            **kwargs,
        )


CONTINUATION_BACKEND_STANDARD = "Standard"
CONTINUATION_BACKEND_COMPATIBILITY = "Compatibility"
CONTINUATION_BACKEND_OPTIONS = (
    CONTINUATION_BACKEND_STANDARD,
    CONTINUATION_BACKEND_COMPATIBILITY,
)

_V36_BALANCED_CONTINUITY = CONTINUITY_OPTIONS[0]
_V36_NON_BALANCED_AUDIO_FALLBACK_REASON = (
    "Masked AV Standard currently requires Balanced 22 when Audio Continuity "
    "is enabled."
)


def _resolve_v36_continuation_transport(
    *,
    continuation_backend,
    audio_continuity,
    continuity,
):
    from ..v2.sequence import (
        MASKED_AV_PREFIX_22_V1,
        MASKED_VIDEO_PREFIX_V1,
        REFERENCE_CONTEXT_V1,
    )

    backend = str(continuation_backend)
    if backend == CONTINUATION_BACKEND_COMPATIBILITY:
        return REFERENCE_CONTEXT_V1, None
    if not bool(audio_continuity):
        return MASKED_VIDEO_PREFIX_V1, None
    if str(continuity) == _V36_BALANCED_CONTINUITY:
        return MASKED_AV_PREFIX_22_V1, None
    return REFERENCE_CONTEXT_V1, _V36_NON_BALANCED_AUDIO_FALLBACK_REASON


def _prepend_v36_transport_report(outputs, fallback_reason):
    if fallback_reason is None:
        return outputs
    status = (
        "Continuation Backend: Standard\n"
        "Resolved transport: Reference Context\n"
        f"Reason: {fallback_reason}\n"
        f"{str(outputs[3])}"
    )
    return (*outputs[:3], status, *outputs[4:])


class H3ContinuumSamplerV36(H3ContinuumSamplerV35):
    """V3.6 sampler with Masked continuation and a legacy fallback."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.6 sampler. Standard continuation preserves prior Video/Audio "
        "inside the next H3 target with native noise masks; Compatibility retains the "
        "V3.5 Reference Context transport. With Audio Continuity enabled, Standard "
        "uses Masked AV for Balanced 22 and safely falls back to Reference Context "
        "for Fast 5, Strong 39, and Auto."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.6",
        "MiniMax H3 masked continuation",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema.get("required", {}))
        required["continuation_backend"] = (
            CONTINUATION_BACKEND_OPTIONS,
            {
                "default": CONTINUATION_BACKEND_STANDARD,
                "display_name": "Continuation Backend",
                "advanced": True,
                "tooltip": (
                    "Standard uses the V3.6 target-preserving continuation path. "
                    "With Audio Continuity enabled, Masked AV currently uses Balanced "
                    "22; Fast 5, Strong 39, and Auto safely use Reference Context. "
                    "Compatibility restores the V3.5 Reference Context path for "
                    "older workflows or comparison. Run Storage identity follows "
                    "the resolved transport before execution begins."
                ),
            },
        )
        schema["required"] = required
        return schema

    def run(
        self,
        *args,
        continuation_backend=CONTINUATION_BACKEND_STANDARD,
        **kwargs,
    ):
        audio_continuity = kwargs.get("audio_continuity")
        if audio_continuity is None:
            audio_continuity = True
        continuity = kwargs.get("continuity", _V36_BALANCED_CONTINUITY)
        transport, fallback_reason = _resolve_v36_continuation_transport(
            continuation_backend=continuation_backend,
            audio_continuity=audio_continuity,
            continuity=continuity,
        )
        kwargs["continuation_transport"] = transport
        outputs = super().run(*args, **kwargs)
        return _prepend_v36_transport_report(outputs, fallback_reason)


class H3ContinuumStillImageGuideV37:
    """One frame-based image anchor for the V3.7 sampler."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "Prepare one Still Image Guide at an absolute 24 fps output frame. "
        "The V3.7 sampler resolves it to the owning physical group and Core frame_idx."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Still Image Guide",
        "MiniMax H3 Add Guide timeline",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "absolute_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "step": 1,
                        "display_name": "Absolute Frame (24 fps)",
                        "tooltip": (
                            "Zero-based frame on the final visible Continuum timeline."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (GUIDE_TYPE,)
    RETURN_NAMES = ("guide",)
    FUNCTION = "pack"
    CATEGORY = CONTINUUM_CATEGORY

    def pack(self, image, absolute_frame):
        return (make_still_image_guide(image, absolute_frame),)


class H3ContinuumSamplerV37(H3ContinuumSamplerV36):
    """V3.6-compatible sampler with one physical-group Still Image Guide."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.7 Stage 2 sampler. It preserves V3.6 behavior and can "
        "inject one Still Image Guide into only its resolved physical sampling group."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.7",
        "MiniMax H3 absolute frame guide",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional["guide"] = (
            GUIDE_TYPE,
            {
                "display_name": "Still Image Guide (Optional)",
                "tooltip": (
                    "Optional V3.7 Still Image Guide. Only the owning physical "
                    "sampling group receives the Core minimax_keyframes entry."
                ),
            },
        )
        schema["optional"] = optional
        return schema

    def run(self, guide=None, **kwargs):
        guide_source = prepare_still_image_guide_source(
            guide,
            output_width=int(kwargs["width"]),
            output_height=int(kwargs["height"]),
        )
        return super().run(guide_source=guide_source, **kwargs)


class H3ContinuumSamplerV38(H3ContinuumSamplerV37):
    """V3.7 Production facade with the shared H3 resolution preset UI."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "H3 Continuum V3.8 facade over the unchanged V3.7 Production engine. "
        "Size Source and preset resolve to the existing width/height runtime contract; "
        "post-generation reliability diagnostics inspect frame/audio grids, Core "
        "compatibility, and relative visual conditioning load without changing tensors."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.8",
        "MiniMax H3 resolution preset",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        resolved_required = {}
        for name, definition in schema.get("required", {}).items():
            if name == "width":
                resolved_required["aspect"] = (
                    H3_ASPECT_OPTIONS,
                    {
                        "default": H3_ASPECT_AUTO,
                        "display_name": "Legacy Aspect",
                        "advanced": True,
                        "tooltip": (
                            "Saved V3.8 workflow compatibility only. The frontend migrates "
                            "this value to Size Source and hides it from the Main UI."
                        ),
                    },
                )
                resolved_required["preset"] = (
                    H3_PRESET_OPTIONS,
                    {
                        "default": H3_PRESET_DRAFT,
                        "display_name": "Resolution Preset",
                        "tooltip": (
                            "Used only with Size Source = First Image. Draft is fastest; "
                            "Balanced retains more detail; Native 768 uses the H3 native "
                            "short edge; Custom uses Custom MP."
                        ),
                    },
                )
                resolved_required["custom_mp"] = (
                    "FLOAT",
                    {
                        "default": H3_CUSTOM_MP_DEFAULT,
                        "min": H3_CUSTOM_MP_MIN,
                        "max": H3_CUSTOM_MP_MAX,
                        "step": 0.01,
                        "display_name": "Custom MP",
                        "tooltip": (
                            "Custom target megapixels while preserving the connected First "
                            "Image aspect ratio. Used only when Resolution Preset is Custom."
                        ),
                    },
                )
                continue
            if name == "height":
                continue
            resolved_required[name] = definition
        # Keep every existing V3.8 widget index stable. Review controls are
        # intentionally appended instead of being inserted near Run Storage.
        resolved_required["generation_mode"] = (
            GENERATION_MODE_OPTIONS,
            {
                "default": GENERATION_MODE_FULL_RUN,
                "display_name": "Generation Mode",
                "tooltip": (
                    "Full Run preserves normal Production execution. Review Each "
                    "Chunk generates at most one new physical group per Queue and "
                    "requires Run Storage = Save + Auto Resume."
                ),
            },
        )
        resolved_required["review_action"] = (
            REVIEW_ACTION_OPTIONS,
            {
                "default": REVIEW_ACTION_CONTINUE,
                "display_name": "Review Action",
                "tooltip": (
                    "Continue / Next accepts the current review and advances. "
                    "Regenerate Current and Finish Remaining are one-shot actions. "
                    "Smart Regenerate requires Regenerate From = Auto."
                ),
            },
        )
        resolved_required["take_group"] = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 16,
                "step": 1,
                "advanced": True,
                "display_name": "Selected Take Group",
                "tooltip": (
                    "Internal Render History selection. Use the visible Previous/Next Take "
                    "controls instead of editing this value directly."
                ),
            },
        )
        resolved_required["take_revision_id"] = (
            "STRING",
            {
                "default": "",
                "advanced": True,
                "display_name": "Selected Take Revision",
                "tooltip": (
                    "Immutable revision selected by Render History. The visible Take controls "
                    "manage this value from verified Run Storage data."
                ),
            },
        )
        resolved_required["take_action"] = (
            TAKE_ACTION_OPTIONS,
            {
                "default": TAKE_ACTION_AUTOMATIC,
                "advanced": True,
                "display_name": "Take Action",
                "tooltip": (
                    "One-shot Render History action. Selecting a Take alone does not change "
                    "the canonical branch; Queue normally after choosing an action."
                ),
            },
        )
        resolved_required["size_source"] = (
            H3_SIZE_SOURCE_OPTIONS,
            {
                "default": H3_SIZE_SOURCE_LEGACY,
                "display_name": "Size Source",
                "tooltip": (
                    "First Image preserves its aspect at the selected Resolution Preset. "
                    "Manual uses Width and Height exactly. Legacy Aspect is accepted only "
                    "for saved-workflow and API compatibility."
                ),
            },
        )
        resolved_required["width"] = (
            "INT",
            {
                "default": H3_MANUAL_WIDTH_DEFAULT,
                "min": H3_CANVAS_MIN,
                "max": H3_CANVAS_MAX,
                "step": H3_CANVAS_MULTIPLE,
                "display_name": "Width",
                "tooltip": (
                    "Exact output width in Manual mode. Use a multiple of 32. Manual mode is "
                    "the normal choice for T2VA or workflows without a First Image."
                ),
            },
        )
        resolved_required["height"] = (
            "INT",
            {
                "default": H3_MANUAL_HEIGHT_DEFAULT,
                "min": H3_CANVAS_MIN,
                "max": H3_CANVAS_MAX,
                "step": H3_CANVAS_MULTIPLE,
                "display_name": "Height",
                "tooltip": (
                    "Exact output height in Manual mode. Use a multiple of 32. Manual mode is "
                    "the normal choice for T2VA or workflows without a First Image."
                ),
            },
        )
        # RefMod (ComfyUI-MiniMaxH3Mod) controls mirror `Apply H3 RefMod` and are
        # appended after the existing V3.8 widgets so saved widget indices stay
        # stable. They only act when the optional `refmods` socket is connected.
        resolved_required.update(cls._refmod_widget_schema())
        schema["required"] = resolved_required
        optional = dict(schema.get("optional", {}))
        optional["audio_references"] = (
            REFERENCE_AUDIOS_TYPE,
            {
                "display_name": "Audio References (Optional)",
                "tooltip": (
                    "Optional ordered bundle from H3 Continuum Reference Audios. "
                    "Do not connect it together with the legacy single Reference Audio input."
                ),
            },
        )
        optional["refmods"] = (
            REFMODS_TYPE,
            {
                "display_name": "RefMods (Optional)",
                "tooltip": (
                    "Optional H3_REF_MODS bundle from the ComfyUI-MiniMaxH3Mod pack "
                    "(Load H3 RefMods, Load H3 RefMod Axis, or Create H3 RefMod). "
                    "The Sampler applies it like Apply H3 RefMod, injecting the "
                    "reference blocks into every chunk. Leave unconnected to disable; "
                    "the RefMod settings below are then ignored."
                ),
            },
        )
        schema["optional"] = optional
        return schema

    @classmethod
    def _refmod_widget_schema(cls):
        """Adjustable settings equivalent to the `Apply H3 RefMod` node."""
        return {
            "refmod_retention": (
                "FLOAT",
                {
                    "default": REFMOD_DEFAULT_RETENTION,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "advanced": True,
                    "display_name": "RefMod Retention",
                    "tooltip": (
                        "Master reference strength multiplied with each RefMod row's "
                        "strength. 1.0 = fully preserved, 0.7 = partially preserved, "
                        "0.4 = attribute transfer, 0.15 = weak reference, 0 = no "
                        "reference. Used only when RefMods is connected."
                    ),
                },
            ),
            "refmod_curve_direction": (
                list(REFMOD_CURVE_DIRECTIONS),
                {
                    "default": REFMOD_DEFAULT_CURVE_DIRECTION,
                    "advanced": True,
                    "display_name": "RefMod Curve Direction",
                    "tooltip": (
                        "Weighting envelope across each mod's OWN reference frames "
                        "(stacked images or video-ref frames), not the output timeline. "
                        "constant keeps every frame at full strength; the concept_at_* "
                        "directions fade part of the stack toward blur. Single-image "
                        "mods ignore the direction and use Curve Value as a strength cap."
                    ),
                },
            ),
            "refmod_curve_shape": (
                list(REFMOD_CURVE_SHAPES),
                {
                    "default": REFMOD_DEFAULT_CURVE_SHAPE,
                    "advanced": True,
                    "display_name": "RefMod Curve Shape",
                    "tooltip": (
                        "How the weighting travels between its endpoints: linear, ease, "
                        "sigmoid, tanh, quadratic, cubic, exponential, stair, elastic, "
                        "bump, or dip. Only matters when Curve Direction is not constant."
                    ),
                },
            ),
            "refmod_curve_value": (
                "FLOAT",
                {
                    "default": REFMOD_DEFAULT_CURVE_VALUE,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "advanced": True,
                    "display_name": "RefMod Curve Value",
                    "tooltip": (
                        "Endpoint weight of the curve: both endpoints for constant and "
                        "concept_at_ends, the start for concept_at_start, the end for "
                        "concept_at_end, the peak for concept_at_middle. On single-image "
                        "mods this is a plain strength cap."
                    ),
                },
            ),
            "refmod_scramble_seed": (
                "INT",
                {
                    "default": REFMOD_DEFAULT_SCRAMBLE_SEED,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "advanced": True,
                    "display_name": "RefMod Scramble Seed",
                    "tooltip": (
                        "-1 = off (all refs in saved order). With 2 or more refs in the "
                        "bundle, a seed of 0 or higher shuffles the order and, in the "
                        "subset modes, keeps a subset so a different ref leads each run. "
                        "The same seed always gives the same scramble."
                    ),
                },
            ),
            "refmod_scramble_mode": (
                list(REFMOD_SCRAMBLE_MODES),
                {
                    "default": REFMOD_DEFAULT_SCRAMBLE_MODE,
                    "advanced": True,
                    "display_name": "RefMod Scramble Mode",
                    "tooltip": (
                        "shuffle keeps every ref in a seeded random order; subset keeps "
                        "only Scramble Keep refs; legacy_subset keeps a seeded random "
                        "half-to-all subset. Ignored while Scramble Seed is -1."
                    ),
                },
            ),
            "refmod_scramble_keep": (
                "INT",
                {
                    "default": REFMOD_DEFAULT_SCRAMBLE_KEEP,
                    "min": 1,
                    "max": 80,
                    "step": 1,
                    "advanced": True,
                    "display_name": "RefMod Scramble Keep",
                    "tooltip": (
                        "Number of refs retained when Scramble Mode is subset. "
                        "shuffle and legacy_subset ignore this value."
                    ),
                },
            ),
            "refmod_max_tokens": (
                "INT",
                {
                    "default": REFMOD_DEFAULT_MAX_TOKENS,
                    "min": 0,
                    "max": 1048576,
                    "step": 1,
                    "advanced": True,
                    "display_name": "RefMod Token Budget",
                    "tooltip": (
                        "Total reference token budget for the bundle after copies and "
                        "scrambling. 0 disables the limit. A bundle above a positive "
                        "budget is rejected before sampling, matching Apply H3 RefMod."
                    ),
                },
            ),
            "refmod_override": (
                "BOOLEAN",
                {
                    "default": False,
                    "advanced": True,
                    "display_name": "RefMod Use Saved Config",
                    "tooltip": (
                        "Use the retention and curve fixed into a mod's own metadata by "
                        "Fix H3 RefMod Config instead of the RefMod widgets above. The "
                        "first mod in the bundle with a saved config wins; without one "
                        "the widgets stay in force and the report notes it."
                    ),
                },
            ),
        }

    @classmethod
    def IS_CHANGED(
        cls,
        generation_mode=GENERATION_MODE_FULL_RUN,
        **kwargs,
    ):
        # Review execution depends on validated Run Storage state that is
        # intentionally outside the visible graph inputs.  Always execute the
        # V3.8 sampler in Review mode and let Run Storage remain the sole
        # authority for exact prefix reuse.  Full Run keeps Core's stable cache
        # behavior, and this method is deliberately not inherited by V3.7/Easy.
        if generation_mode == GENERATION_MODE_REVIEW:
            return float("NaN")
        return False

    def run(
        self,
        aspect=H3_ASPECT_AUTO,
        preset=H3_PRESET_DRAFT,
        custom_mp=H3_CUSTOM_MP_DEFAULT,
        generation_mode=GENERATION_MODE_FULL_RUN,
        review_action=REVIEW_ACTION_CONTINUE,
        take_group=0,
        take_revision_id="",
        take_action=TAKE_ACTION_AUTOMATIC,
        size_source=H3_SIZE_SOURCE_LEGACY,
        width=H3_MANUAL_WIDTH_DEFAULT,
        height=H3_MANUAL_HEIGHT_DEFAULT,
        refmods=None,
        refmod_retention=REFMOD_DEFAULT_RETENTION,
        refmod_curve_direction=REFMOD_DEFAULT_CURVE_DIRECTION,
        refmod_curve_shape=REFMOD_DEFAULT_CURVE_SHAPE,
        refmod_curve_value=REFMOD_DEFAULT_CURVE_VALUE,
        refmod_scramble_seed=REFMOD_DEFAULT_SCRAMBLE_SEED,
        refmod_scramble_mode=REFMOD_DEFAULT_SCRAMBLE_MODE,
        refmod_scramble_keep=REFMOD_DEFAULT_SCRAMBLE_KEEP,
        refmod_max_tokens=REFMOD_DEFAULT_MAX_TOKENS,
        refmod_override=False,
        **kwargs,
    ):
        resolution = resolve_h3_size_source(
            size_source=size_source,
            width=width,
            height=height,
            aspect=aspect,
            preset=preset,
            custom_mp=custom_mp,
            first_frame=kwargs.get("first_frame"),
        )
        refmod_status = None
        if refmods is not None:
            # Resolve the bundle exactly like Apply H3 RefMod and scope the
            # injection to a cloned MODEL so every chunk's guider receives the
            # same reference blocks. A disconnected socket is a pure no-op.
            settings = normalize_refmod_settings(
                retention=refmod_retention,
                curve_direction=refmod_curve_direction,
                curve_shape=refmod_curve_shape,
                curve_value=refmod_curve_value,
                scramble_seed=refmod_scramble_seed,
                scramble_mode=refmod_scramble_mode,
                scramble_keep=refmod_scramble_keep,
                max_total_tokens=refmod_max_tokens,
                override=refmod_override,
            )
            blocks, summary = build_refmod_blocks(refmods, settings)
            if blocks and "model" in kwargs:
                kwargs = dict(kwargs)
                kwargs["model"] = attach_refmod_blocks(kwargs["model"], blocks)
            refmod_status = format_refmod_status(summary)
            print("[H3 Continuum] " + refmod_status.replace("\n", " | "))
        outputs = super().run(
            width=resolution.width,
            height=resolution.height,
            generation_mode=generation_mode,
            review_action=review_action,
            take_group=take_group,
            take_revision_id=take_revision_id,
            take_action=take_action,
            reference_encode_cache=True,
            **kwargs,
        )
        # Preserve non-runtime test doubles and any future non-standard facade
        # result rather than turning diagnostics into an execution requirement.
        if not isinstance(outputs, tuple) or len(outputs) < 4:
            return outputs
        if refmod_status is not None:
            status = str(outputs[3]).rstrip() + "\n" + refmod_status
            outputs = (*outputs[:3], status, *outputs[4:])
        if size_source != H3_SIZE_SOURCE_LEGACY:
            resolution_lines = [
                (
                    f"Resolution: {resolution.width} x {resolution.height} "
                    f"({resolution.actual_mp:.2f} MP); source={resolution.aspect_source}."
                )
            ]
            resolution_lines.extend(resolution.warnings)
            status = str(outputs[3]).rstrip() + "\n" + "\n".join(resolution_lines)
            outputs = (*outputs[:3], status, *outputs[4:])
        diagnostics_mode = str(kwargs.get("diagnostics", "Basic"))
        normalized_diagnostics_mode = normalize_diagnostics_mode(diagnostics_mode)
        if normalized_diagnostics_mode == DIAGNOSTICS_OFF:
            return outputs
        try:
            diagnostics = build_v38_diagnostics(
                video_latents=outputs[0],
                audio_latents=outputs[1],
                assembly_plan=outputs[2],
                output_width=resolution.width,
                output_height=resolution.height,
                chunk_seconds=float(kwargs["chunk_seconds"]),
                reference_images=(
                    kwargs.get("reference_image_1"),
                    kwargs.get("reference_image_2"),
                    kwargs.get("reference_image_3"),
                ),
                reference_size=str(kwargs.get("reference_size", "Match Output")),
                video_guide=kwargs.get("reference_video_1"),
                video_guide_size=str(
                    kwargs.get(
                        "video_reference_size",
                        REFERENCE_VIDEO_SIZE_EFFICIENT,
                    )
                ),
                still_guide_active=kwargs.get("guide") is not None,
            )
            status = append_v38_status(
                outputs[3],
                diagnostics,
                mode=diagnostics_mode,
            )
        except Exception as exc:
            # Generation already succeeded. Reliability reporting must never
            # discard or replace a valid Production result.
            status = str(outputs[3])
            message = (
                "V3.8 Reliability\n"
                "Reliability diagnostics unavailable; generation result was preserved."
            )
            if normalized_diagnostics_mode == DIAGNOSTICS_FULL:
                message += f" ({type(exc).__name__}: {exc})"
            status = status.rstrip() + "\n" + message
        return (*outputs[:3], status, *outputs[4:])


class H3ContinuumMemoryActionPolicyExperimental:
    """Explicit opt-in Reference sizing policy for the A7b experiment."""

    DEPRECATED = False
    CATEGORY = f"{CONTINUUM_CATEGORY}/Advanced"
    DESCRIPTION = (
        "Experimental A7b policy. It may reduce connected Reference Image and "
        "Video Guide sizing by one existing preset step. Automatic MODEL unload "
        "and attention/backend switching are advisory only and are never executed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_action": (
                    ACTION_OPTIONS,
                    {
                        "default": ACTION_DISABLED,
                        "display_name": "Reference Action",
                        "tooltip": (
                            "Disabled is a complete no-op. Reduce References One "
                            "Step applies only to connected Reference inputs and "
                            "reuses their existing Production size presets."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = (MEMORY_ACTION_POLICY_TYPE,)
    RETURN_NAMES = ("memory_action_policy",)
    FUNCTION = "build"

    def build(self, reference_action=ACTION_DISABLED):
        return (make_memory_action_policy(reference_action),)


class H3ContinuumSamplerV38MemoryPolicyExperimental(H3ContinuumSamplerV38):
    """V3.8 Experimental facade with one optional A7b policy input."""

    DEPRECATED = False
    CATEGORY = f"{CONTINUUM_CATEGORY}/Advanced"
    DESCRIPTION = (
        "Experimental V3.8 facade for the explicit A7b Reference size action. "
        "With the policy disconnected or Disabled, it delegates unchanged to "
        "H3 Continuum Sampler V3.8."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Sampler V3.8 Memory Policy Experimental",
        "MiniMax H3 A7b",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional["memory_action_policy"] = (
            MEMORY_ACTION_POLICY_TYPE,
            {
                "tooltip": (
                    "Optional explicit A7b policy. No actual-row trigger, MODEL "
                    "unload, or attention/backend switch is performed."
                )
            },
        )
        schema["optional"] = optional
        return schema

    def run(self, memory_action_policy=None, **kwargs):
        decision = resolve_memory_action_policy(
            memory_action_policy,
            image_mode=str(kwargs.get("reference_size", "Match Output")),
            video_mode=str(
                kwargs.get(
                    "video_reference_size",
                    REFERENCE_VIDEO_SIZE_EFFICIENT,
                )
            ),
            has_reference_images=any(
                kwargs.get(name) is not None
                for name in (
                    "reference_image_1",
                    "reference_image_2",
                    "reference_image_3",
                )
            ),
            has_video_guide=kwargs.get("reference_video_1") is not None,
        )
        execution_kwargs = kwargs
        if decision.enabled:
            execution_kwargs = dict(kwargs)
            execution_kwargs["reference_size"] = decision.effective_image_mode
            execution_kwargs["video_reference_size"] = decision.effective_video_mode
        outputs = super().run(**execution_kwargs)
        policy_status = format_memory_action_status(decision)
        if (
            policy_status is None
            or not isinstance(outputs, tuple)
            or len(outputs) < 4
        ):
            return outputs
        status = str(outputs[3]).rstrip() + "\n" + policy_status
        return (*outputs[:3], status, *outputs[4:])


class H3ContinuumAssembleSeamV34(H3ContinuumAssembleSeamExperimental):
    """Assemble decoded chunks and select original Driving Audio when connected."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "V3.4 decoded-chunk assembler. Driving Audio bypasses generated audio "
        "and Audio Seam while video seam correction remains available."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional["driving_audio"] = (
            "AUDIO",
            {
                "tooltip": (
                    "Connect the sampler Driving Audio output. When present, generated "
                    "audio and Audio Seam are bypassed."
                )
            },
        )
        schema["optional"] = optional
        return schema

    def assemble(self, *args, driving_audio=None, **kwargs):
        preserved_audio = _driving_audio_from_plan(args, kwargs)
        images, audio, report = super().assemble(*args, **kwargs)
        selected = preserved_audio or _copy_audio(driving_audio)
        if selected is None:
            return images, audio, report

        plan_value = kwargs.get("assembly_plan")
        if plan_value is None and len(args) >= 3:
            plan_value = args[2]
        while isinstance(plan_value, list):
            if len(plan_value) != 1:
                raise ValueError("assembly_plan must contain exactly one value")
            plan_value = plan_value[0]
        exact = kwargs.get("exact_total_duration")
        if exact is None and len(args) >= 4:
            exact = args[3]
        while isinstance(exact, list):
            if len(exact) != 1:
                raise ValueError("exact_total_duration must contain exactly one value")
            exact = exact[0]
        if bool(exact):
            _, selected, _ = enforce_total_frames(
                images,
                selected,
                target_frames=int(plan_value["target_frames"]),
                preserve_final_frame=bool(plan_value.get("preserve_final_frame", False)),
            )
        source = "assembly plan" if preserved_audio is not None else "direct input"
        samples = int(selected["waveform"].shape[-1])
        report = str(report) + (
            f"\nDriving Audio: preserved source selected from {source}; "
            f"sample_rate={selected['sample_rate']}, samples={samples}; "
            "generated audio and Audio Seam bypassed."
        )
        return images, selected, report


def _assembly_argument(args, kwargs, name, index):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def _insert_projection_report(report, line):
    lines = str(report).splitlines()
    for index, value in enumerate(lines):
        if value.startswith("Memory [assemble preflight]"):
            lines.insert(index + 1, line)
            return "\n".join(lines)
    lines.append(line)
    return "\n".join(lines)


class H3ContinuumAssembleSeamV35(H3ContinuumAssembleSeamV34):
    """Independent V3.5 direct-write assembly with selectable IMAGE storage."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "V3.5 decoded-chunk assembler with Auto, RAM, or Disk-backed IMAGE "
        "storage, direct Exact Duration writes, and small-patch Video Seam."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        diagnostics = required.pop("diagnostics")
        required["buffer_backend"] = (
            BUFFER_BACKEND_OPTIONS,
            {
                "default": BUFFER_BACKEND_AUTO,
                "display_name": "Buffer Backend",
                "tooltip": (
                    "Auto keeps outputs up to 4 GiB in RAM only when physical "
                    "memory has a safety reserve; otherwise it maps the final "
                    "IMAGE to disk. Manual RAM and Disk-backed remain available."
                ),
            },
        )
        required["diagnostics"] = diagnostics
        schema["required"] = required
        return schema

    def assemble(
        self,
        images,
        audio,
        assembly_plan,
        exact_total_duration,
        audio_seam,
        video_seam,
        buffer_backend,
        diagnostics,
        driving_audio=None,
    ):
        image_chunks = _chunk_list(images)
        audio_chunks = _chunk_list(audio)
        plan = _singleton(assembly_plan, "assembly_plan")
        exact = bool(_singleton(exact_total_duration, "exact_total_duration"))
        audio_seam_mode = str(_singleton(audio_seam, "audio_seam"))
        video_seam_mode = str(_singleton(video_seam, "video_seam"))
        backend = str(_singleton(buffer_backend, "buffer_backend"))
        diagnostics_mode = str(_singleton(diagnostics, "diagnostics"))
        preserved_audio = _copy_audio(plan.get(_DRIVING_AUDIO_PLAN_KEY))
        selected_audio = preserved_audio or _copy_audio(driving_audio)
        selected_audio_source = None
        if selected_audio is not None:
            selected_audio_source = (
                "assembly plan" if preserved_audio is not None else "direct input"
            )
        projection_line = None
        if diagnostics_is_full(diagnostics_mode):
            try:
                projection = project_assembly_buffers(
                    images=image_chunks,
                    audio=audio_chunks,
                    assembly_plan=plan,
                    exact_total_duration=exact,
                )
                projection_line = format_assembly_projection(projection) + (
                    "; V3.5 Exact Duration IMAGE write=direct; "
                    "additional final IMAGE copy=no"
                )
            except Exception as exc:
                projection_line = (
                    "Projected assembly buffers [V3.5]: unavailable "
                    f"({type(exc).__name__}: {exc})"
                )

        result_images, result_audio, report = assemble_decoded_chunks_v35(
            images=image_chunks,
            audio=audio_chunks,
            assembly_plan=plan,
            exact_total_duration=exact,
            audio_seam=audio_seam_mode,
            video_seam=video_seam_mode,
            diagnostics=diagnostics_mode,
            buffer_backend=backend,
            final_audio_override=selected_audio,
            final_audio_source=selected_audio_source,
        )
        if projection_line is not None:
            report = _insert_projection_report(report, projection_line)
        return result_images, result_audio, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumSamplerV34": H3ContinuumSamplerV34,
    "H3ContinuumSamplerV35": H3ContinuumSamplerV35,
    "H3ContinuumSamplerV36": H3ContinuumSamplerV36,
    "H3ContinuumStillImageGuideV37": H3ContinuumStillImageGuideV37,
    "H3ContinuumSamplerV37": H3ContinuumSamplerV37,
    "H3ContinuumSamplerV38": H3ContinuumSamplerV38,
    "H3ContinuumReferenceAudios": H3ContinuumReferenceAudios,
    "H3ContinuumMemoryActionPolicyExperimental": (
        H3ContinuumMemoryActionPolicyExperimental
    ),
    "H3ContinuumSamplerV38MemoryPolicyExperimental": (
        H3ContinuumSamplerV38MemoryPolicyExperimental
    ),
    "H3ContinuumAssembleSeamV34": H3ContinuumAssembleSeamV34,
    "H3ContinuumAssembleSeamV35": H3ContinuumAssembleSeamV35,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumSamplerV34": "H3 Continuum Sampler V3.4",
    "H3ContinuumSamplerV35": "H3 Continuum Sampler V3.5",
    "H3ContinuumSamplerV36": "H3 Continuum Sampler V3.6",
    "H3ContinuumStillImageGuideV37": "H3 Continuum Still Image Guide V3.7",
    "H3ContinuumSamplerV37": "H3 Continuum Sampler V3.7",
    "H3ContinuumSamplerV38": "H3 Continuum Sampler V3.8",
    "H3ContinuumReferenceAudios": "H3 Continuum Reference Audios",
    "H3ContinuumMemoryActionPolicyExperimental": (
        "H3 Continuum Memory Action Policy (Experimental)"
    ),
    "H3ContinuumSamplerV38MemoryPolicyExperimental": (
        "H3 Continuum Sampler V3.8 Memory Policy (Experimental)"
    ),
    "H3ContinuumAssembleSeamV34": "H3 Continuum Assemble + Seam V3.4",
    "H3ContinuumAssembleSeamV35": "H3 Continuum Finalize",
}
