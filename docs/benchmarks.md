# Benchmarks

Performance comparisons between original imgaug and imgaug2.

!!! warning "Benchmark numbers are not universal"
    Any timing numbers depend heavily on:

    - CPU model + thermals
    - OpenCV build + thread settings
    - NumPy/SciPy versions
    - batch size and image resolution

    Treat published numbers as **examples**, not guarantees.
    Always run the benchmark suite on your target hardware.

## Test Configuration

- **Batch**: 16 images
- **Image size**: 256×256×3 (RGB)
- **Iterations**: 100 per augmenter

## imgaug vs imgaug2 Comparison

Example results on 256×256 images (16 image batch).
Use these as a rough sanity check, not as an expectation.

| Augmenter | Original imgaug | imgaug2 | Speedup |
|-----------|-----------------|---------|---------|
| **GaussianBlur** | 5.52s | 0.25s | **~22×** |
| **ElasticTransformation** | 26.31s | 0.86s | **~30×** |
| **AffineAll** | 212.92s | 1.28s | **~166×** |
| Crop-percent | 0.81s | 0.43s | ~2× |
| Fliplr | 0.02s | 0.01s | ~2× |
| Identity | 0.05s | 0.03s | ~2× |

### Key Improvements

The massive speedups in geometric augmenters (Affine, ElasticTransformation) come from:

- Modern NumPy/SciPy optimizations
- Removal of legacy compatibility overhead
- Updated OpenCV bindings

## Original imgaug Baseline

These values are preserved from the original imgaug repository for reference.

### Images (16× 256×256×3)

| Augmenter | SUM | Avg/iter |
|-----------|-----|----------|
| Identity | 0.053s | 0.0005s |
| Crop-px | 0.819s | 0.0082s |
| Crop-percent | 0.808s | 0.0081s |
| Fliplr | 0.024s | 0.0002s |
| Flipud | 0.024s | 0.0002s |
| GaussianBlur | 5.517s | 0.0552s |
| AdditiveGaussianNoise | 1.036s | 0.0104s |
| Dropout | 0.411s | 0.0041s |
| Multiply | 0.194s | 0.0019s |
| ContrastNormalization | 0.353s | 0.0035s |
| Grayscale | 0.238s | 0.0024s |
| ElasticTransformation | 26.305s | 0.2631s |
| AffineOrder0 | 1.803s | 0.0180s |
| AffineOrder1 | 2.206s | 0.0221s |
| AffineAll | 212.918s | 2.1292s |

## Platform Notes

The benchmark runner stores basic system metadata (OS/CPU/RAM) with each run so
you can compare results across machines. Use `--label ...` to keep runs organized.

## Running Your Own Benchmarks

### Full benchmark suite (recommended)

Run everything (CPU suite + legacy suite + ops suite + report generation):

```bash
python -m benchmarks.run_all
```

Useful flags:

- Skip ops-level suite: `--skip-ops`
- Skip legacy suite: `--skip-legacy`
- Skip third-party baseline: `--skip-third-party`

Outputs:

- JSON results: `benchmarks/results/*.json`
- Markdown report: `benchmarks/reports/out/benchmark_report.md`

### Benchmarks runner (advanced)

Run the augmenter-level benchmark suite only (writes JSON with `system_info` to `benchmarks/results/`):

```bash
python -m benchmarks.runner --platform cpu
```

Generate a Markdown summary report from existing JSON results:

```bash
python -m benchmarks.reports.generate_report --results-dir benchmarks/results
```

Tips:

- Filter augmenters: `--augmenters Identity,GaussianBlur_small`
- Filter configs: `--configs 16x256x256x3`
- Add a label (useful for CI/hardware tracking): `--label "m3pro-2025-12"`

### Legacy performance check (historical)

```bash
cd imgaug2
python checks/check_performance.py
```

Or benchmark specific augmenters:

```python
import time
import numpy as np
import imgaug2.augmenters as iaa

# Setup
images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
          for _ in range(16)]

aug = iaa.GaussianBlur(sigma=(0.5, 1.5))

# Benchmark
start = time.perf_counter()
for _ in range(100):
    aug(images=images)
elapsed = time.perf_counter() - start

print(f"Total: {elapsed:.3f}s")
print(f"Per batch: {elapsed/100*1000:.2f}ms")
print(f"Per image: {elapsed/1600*1000:.2f}ms")
```

## Contributing Benchmarks

We welcome benchmark contributions on different hardware:

1. Run `checks/check_performance.py`
2. Note your hardware specs (CPU, RAM, OS)
3. Submit results via GitHub issue or PR

Hardware we're looking for:
- Apple Silicon (M1, M2, M3, M4)
- AMD CPUs (Ryzen, EPYC)
- Intel 12th/13th/14th gen
