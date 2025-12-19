# imgaug2 Benchmarks

Performance benchmarking suite for imgaug2 augmentation pipelines.

## Quick Start

```bash
# Run the full benchmark suite (after pip install)
imgaug2-bench

# Or run as a module
python -m benchmarks.run_all

# Run only augmenter-level benchmarks
python -m benchmarks.runner --platform cpu

# Run operations-level benchmarks (CPU vs MLX)
python -m benchmarks.ops
```

## Installation

The benchmark CLI is installed with the package:

```bash
pip install -e .
```

This provides the `imgaug2-bench` command.

## Directory Structure

```
benchmarks/
├── runner.py              # Main augmenter-level benchmark runner
├── ops.py                 # Operations-level benchmarks (CPU vs MLX)
├── legacy.py              # Legacy benchmarks (parity with checks/)
├── run_all.py             # Master entrypoint for all suites
├── config.py              # Benchmark configurations
├── third_party_baseline.py # Albumentations comparison (optional)
├── platforms/             # Platform-specific helpers
│   ├── cpu.py             # Core timing and stats collection
│   ├── apple_silicon.py   # Apple Silicon metadata
│   └── nvidia_cuda.py     # NVIDIA GPU metadata
├── reports/
│   └── generate_report.py # JSON to Markdown report generator
└── results/               # Output directory for benchmark results
```

## Benchmark Suites

### 1. Augmenter-Level Benchmarks (`runner.py`)

Tests high-level augmenters with realistic batch processing across multiple image configurations.

```bash
# Run all augmenters
python -m benchmarks.runner --platform cpu

# Run specific augmenters
python -m benchmarks.runner --augmenters "Fliplr,GaussianBlur_small,Affine_rotate"

# Run specific image configs
python -m benchmarks.runner --configs "16x256x256x3,32x256x256x3"

# List available options
python -m benchmarks.runner --list-augmenters
python -m benchmarks.runner --list-configs
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--platform` | `cpu` | Platform for metadata (cpu, apple, nvidia) |
| `--output` | `benchmarks/results` | Output directory |
| `--iterations` | `100` | Timing iterations per benchmark |
| `--warmup` | `5` | Warm-up iterations |
| `--seed` | `42` | Random seed |
| `--augmenters` | all | Comma-separated augmenter names |
| `--configs` | all | Comma-separated config keys |
| `--label` | `""` | Custom label for results file |
| `--fail-fast` | `False` | Stop on first error |

### 2. Operations-Level Benchmarks (`ops.py`)

Benchmarks low-level operations comparing CPU (OpenCV) vs MLX implementations.

```bash
python -m benchmarks.ops --output benchmarks/results
```

Tests three variants for each operation:

- **CPU baseline**: cv2 implementation
- **MLX roundtrip**: NumPy → MLX → NumPy per call
- **MLX device**: Data stays on MLX device between calls

Note: Some ops exposed under `imgaug2.mlx` (e.g. `jpeg_compression`) require a CPU codec and will always roundtrip through host memory; these are benchmarked as roundtrip-only.

Operations tested:

- `gaussian_blur`
- `affine_transform`
- `perspective_transform`
- `pipeline` (blur → affine → add)
- `pose_presets` (6 presets for pose estimation)
- `mlx_op_coverage` (MLX-only coverage for additional ops at a small 1x128x128 config)

You can run only the MLX coverage slice:

```bash
python -m benchmarks.ops --only mlx_op_coverage --output benchmarks/results
```

### 3. Legacy Benchmarks (`legacy.py`)

Maintains compatibility with `checks/check_performance.py` for historical comparison.

```bash
python -m benchmarks.legacy --output benchmarks/results
```

### 4. Third-Party Baseline (`third_party_baseline.py`)

Compares imgaug2 against Albumentations.

```bash
# Requires: pip install albumentations
python -m benchmarks.third_party_baseline --output benchmarks/results
```

## Full Suite

Run all benchmarks and generate a unified report:

```bash
python -m benchmarks.run_all

# With options
python -m benchmarks.run_all \
  --output benchmarks/results \
  --report benchmarks/reports/out/benchmark_report.md \
  --iterations 100 \
  --skip-legacy \
  --skip-third-party
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `benchmarks/results` | Results output directory |
| `--report` | `benchmarks/reports/out/benchmark_report.md` | Report output path |
| `--iterations` | `100` | Timing iterations |
| `--warmup` | `5` | Warm-up iterations |
| `--skip-legacy` | `False` | Skip legacy suite |
| `--skip-ops` | `False` | Skip ops suite |
| `--skip-third-party` | `False` | Skip third-party baseline |

## Image Configurations

Default configurations defined in `config.py`:

| Key | Shape | Description |
|-----|-------|-------------|
| `1x64x64x3` | (1, 64, 64, 3) | Single small image |
| `1x256x256x3` | (1, 256, 256, 3) | Single medium image |
| `16x256x256x3` | (16, 256, 256, 3) | Standard batch |
| `32x256x256x3` | (32, 256, 256, 3) | Large batch |

## Output Format

### JSON Results

Results are saved as JSON with the following structure:

```json
{
  "system_info": {
    "platform": "Darwin",
    "architecture": "arm64",
    "python_version": "3.12.x",
    "numpy_version": "1.x.x",
    "imgaug2_version": "2.x.x",
    "timestamp": "2025-12-18T..."
  },
  "benchmarks": {
    "AugmenterName": {
      "16x256x256x3": {
        "iterations": 100,
        "total_time_s": 0.668,
        "avg_time_s": 0.00668,
        "min_time_s": 0.00546,
        "max_time_s": 0.03636,
        "std_time_s": 0.00360,
        "p50_time_s": 0.00608,
        "p95_time_s": 0.00680,
        "images_per_sec": 2392.2,
        "memory_delta_mb": 0.328,
        "rss_high_water_delta_mb": 0.328
      }
    }
  }
}
```

### Metrics

- **Timing**: total, avg, min, max, std, p50, p95 (seconds)
- **Throughput**: images/sec
- **Memory**: `memory_delta_mb` is a peak-RSS (high-water) delta; `rss_high_water_delta_mb` mirrors it explicitly

### Generating Reports

Convert JSON results to Markdown:

```bash
python -m benchmarks.reports.generate_report \
  --results-dir benchmarks/results \
  --output benchmarks/reports/out/benchmark_report.md
```

## Available Augmenters

The benchmark suite includes 40+ augmenters:

**Fast Operations:**
- Identity, Fliplr, Flipud, Rot90

**Geometric:**
- Affine (rotate, scale, combined), PerspectiveTransform
- ElasticTransformation, PiecewiseAffine

**Blur:**
- GaussianBlur, AverageBlur, MedianBlur, MotionBlur, BilateralBlur

**Noise:**
- AdditiveGaussianNoise, Dropout, CoarseDropout, SaltAndPepper

**Color:**
- Grayscale, AddToHueAndSaturation, Multiply, LinearContrast, CLAHE

**Segmentation:**
- Superpixels, Voronoi

**Pipelines:**
- Light, Medium, Heavy complexity presets

Run `python -m benchmarks.runner --list-augmenters` for the full list.

## Benchmark Analysis (`analyze.py`)

Analyzes benchmark results to determine optimal backend routing.

```bash
# Text output (summary)
python -m benchmarks.analyze benchmarks/results/ops_*.json

# Markdown output (for docs)
python -m benchmarks.analyze benchmarks/results/ops_*.json --markdown

# JSON output (for programmatic use)
python -m benchmarks.analyze benchmarks/results/ops_*.json --json

# Write to file
python -m benchmarks.analyze results/*.json --markdown --output analysis.md
```

### Analysis Features

1. **Break-even detection**: Finds the batch size and image size where MLX becomes faster
2. **Outlier flagging**: Detects anomalies like unexpected performance drops
3. **Routing rules generation**: Produces ready-to-use routing thresholds

### Example Output

```markdown
## Break-Even Points

| Op | Mode | Recommendation |
|:---|:-----|:---------------|
| affine_transform | device | Use MLX at B>=1 when H*W>=147,456; Use MLX at B>=2 when H*W>=65,536 |
| gaussian_blur | device | Use MLX at B>=1 when H*W>=262,144; Use MLX at B>=2 when H*W>=65,536 |
| perspective_transform | device | Use MLX at B>=1 when H*W>=65,536; Use MLX at B>=2 when H*W>=16,384 |
```

## Backend Routing (`imgaug2.mlx.router`)

Based on benchmark analysis, imgaug2 includes a shape-aware backend router:

```python
from imgaug2.mlx import should_use_mlx, get_backend, get_routing_info

# Check if MLX should be used
if should_use_mlx("affine_transform", batch=16, height=256, width=256):
    # Use MLX implementation
    pass

# Get backend recommendation
backend = get_backend("gaussian_blur", batch=1, height=512, width=512)
# Returns: "mlx" or "cpu"

# Get detailed routing info
info = get_routing_info("affine_transform")
# {'op': 'affine_transform', 'category': 'geometric', 'min_total_pixels': 65536, ...}
```

### Routing Rules Summary

| Operation | Use MLX when... |
|:----------|:----------------|
| `affine_transform` | B>=2 and H*W>=65,536, or B>=1 and H*W>=147,456 |
| `perspective_transform` | B>=2 and H*W>=16,384, or B>=1 and H*W>=65,536 |
| `gaussian_blur` | B>=2 and H*W>=65,536, or B>=1 and H*W>=262,144 |
| `fliplr/flipud/rot90` | Always (negligible overhead) |

For small images (64x64) or batch=1 at small sizes, CPU often wins due to MLX kernel launch overhead.

## Notes

- **CPU-based**: Computation runs on CPU; platform selection affects only metadata collection
- **MLX optional**: MLX benchmarks require Apple Silicon
- **Reproducible**: Uses deterministic seeding for consistent results
- **Memory tracking**: Uses `resource.getrusage()` (best on macOS/Linux)
- **Progress**: Uses tqdm if available, falls back to heartbeat printing
