# overlay (deprecated)

`imgaug2.augmenters.overlay` exists for backwards compatibility with older `imgaug`
code. It is a **deprecated alias** for the blending module:

- Use `imgaug2.augmenters.blend` (recommended)
- The primary public classes are `BlendAlpha*` (see `docs/augmenters/blend.md`)

## Legacy Name Mapping

Older code may use these names (from the deprecated overlay API):

- `overlay.Alpha` → `blend.BlendAlpha`
- `overlay.AlphaElementwise` → `blend.BlendAlphaElementwise`
- `overlay.SimplexNoiseAlpha` → `blend.BlendAlphaSimplexNoise`
- `overlay.FrequencyNoiseAlpha` → `blend.BlendAlphaFrequencyNoise`

If you see parameter names `first`/`second` in very old code, imgaug2 uses
`foreground`/`background` instead (matching the newer API naming).

