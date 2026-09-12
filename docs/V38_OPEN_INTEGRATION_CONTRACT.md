# V3.8 Open Integration Contract

This document defines the supported boundary between H3 Continuum V3.8 and
external processing, decoder, and acceleration nodes. It does not add a public
node or change the Sampling, State, Session, Run Storage, or Assembly Plan
schemas.

## Supported pipeline

```text
H3 Continuum Sampler V3.8
  -> external latent processor or latent upscaler (optional)
  -> H3 Continuum Second Pass
  -> Core Video/Audio Decode
  -> H3 Continuum Finalize
  -> Core Save Video
```

Pixel upscalers remain outside the latent boundary. If a workflow decodes to
pixels for processing, it must encode the result back to one H3 video LATENT per
physical group before Second Pass.

## External processor contract

An external processor may change only the spatial geometry of the video
latents. It must preserve:

- physical-group count and order;
- one indivisible entry for every Long Terminal Merge group;
- video batch, 24 channels, and temporal latent length;
- one common target H/W across every physical group;
- the matching first-pass audio latent list and its order;
- the original Assembly Plan input.

Spatial video dimensions may be preserved or enlarged. Smaller dimensions,
NaN/Inf values, changed B/C/T geometry, a changed group count/order, or a changed
audio shape are rejected before Second Pass Sampling.

Second Pass rebuilds supported conditioning for the target geometry, samples
complete physical groups, and returns a copied Assembly Plan with only target
geometry and Second Pass execution metadata added. It does not mutate the input
plan or Run Storage. The accepted first-pass audio LATENT objects are returned
unchanged; temporary audio produced during video refinement is discarded.

## Decoder boundary

H3 Continuum has no decoder allowlist. Any Core or third-party Video/Audio VAE
path, including a TensorRT-backed VAE, is compatible when it implements the
public ComfyUI contracts and produces the normal decoded types consumed by
Finalize:

- `IMAGE`: `[frames, height, width, channels]` per physical group;
- `AUDIO`: a mapping with `waveform` and positive `sample_rate` per physical
  group.

Finalize depends on those public decoded types and the Assembly Plan, not on a
decoder class name. Decoder fallback, tiling, device placement, and acceleration
are owned by the decoder or ComfyUI Core.

## Acceleration boundary

SageAttention, Sol-Attn, Spectrum, and comparable accelerators remain external
MODEL wrappers. Continuum uses ComfyUI's public ModelPatcher wrapper APIs,
preserves existing wrapper/model options on call-local clones, and publishes
only the documented `h3_continuum` interoperability hint when continuation
context exists.

Continuum does not select an attention backend, import accelerator-private
functions, or globally monkey-patch an accelerator. Missing optional accelerator
markers are informational and never become package dependencies.

## RefMod boundary

The external ComfyUI-MiniMaxH3Mod pack ("RefMod") is integrated through one
optional `H3_REF_MODS` socket on the V3.8 Sampler plus Advanced controls that
mirror its `Apply H3 RefMod` node. The contract is:

- Continuum has no import-time dependency on the pack. The installed pack is
  discovered lazily by capability only when a bundle is connected; when its
  block builder is present it is used verbatim, otherwise a duck-typed port
  with the same retention, curve, scramble, budget, and saved-config
  semantics builds the blocks from the mod objects.
- Injection is model-scoped: one keyed `OUTER_SAMPLE` wrapper on a call-local
  MODEL clone appends the native reference blocks to the guider conditioning
  of every physical sampling group, after Continuum's own continuation context
  reference. Existing wrappers (Sage, Sol, Spectrum, the pack's `Step Curve`)
  are preserved; the guider conditioning is restored after each call.
- Conditioning construction, continuation transport, Audio, Seed, SIGMAS, Run
  Storage identity, and captured Second Pass refine context are unchanged. A
  disconnected socket is bit-exact with the previous Sampler.
- A bundle above a positive token budget is rejected before sampling exactly
  as the pack rejects it; every other RefMod setting problem is a diagnostic
  note in the status report, never a stop.

## Ownership summary

| Owner | Responsibility |
|---|---|
| RefMod pack | Mod extraction, storage, loading, strength math, and the `H3_REF_MODS` bundle; Continuum only consumes the bundle |
| Continuum | Physical-group identity/order, temporal ownership, conditioning reconstruction, Second Pass validation/Sampling, first-pass audio passthrough, target-geometry plan copy, Finalize assembly |
| External processor | Spatial Video LATENT transformation while preserving B/C/T, group order, and finite values |
| Core or decoder node | Video/Audio VAE decode/encode behavior, tiling, fallback, device lifecycle |
| Accelerator wrapper | Sage/Sol/Spectrum backend selection and wrapper lifecycle |
| User workflow | Correctly wire the matching Video LATENT list, first-pass Audio LATENT list, and original Assembly Plan |

The contract deliberately avoids an All-in-One node. New processors and
decoders can integrate through public ComfyUI types without requiring a
Continuum release, provided these invariants remain true.
