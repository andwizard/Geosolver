# Benchmark data

The tasks in this project are built on eight third-party geospatial benchmark
sources:

| Directory           | Source          |
| ------------------- | --------------- |
| `GeoBenchX/`        | GeoBenchX       |
| `GeoAnalystBench/`  | GeoAnalystBench |
| `GeoColabs/`        | GeoColabs       |
| `GeoJsonAgent/`     | GeoJsonAgent    |
| `Indoor/`           | Indoor          |
| `Location/`         | Location        |
| `Mining/`           | Mining          |
| `PoiMSN/`           | PoiMSN          |

## What is included in the repository

- `datasets.json` — benchmark task metadata (task text, inputs, expected outputs).
  It is included so the benchmark definition is versioned with the code.

## What is not included

- `Data/<source>/` — the geographic data files themselves (shapefiles,
  GeoTIFFs, GeoJSON, etc.). They total several GB and are excluded via
  `.gitignore`. Obtain them from each benchmark's original distribution.
- Reference solutions (`Output/solutions/`) are excluded from the repository.
  The `Output/` directory itself is kept (tracked via `Output/.gitkeep`);
  generated runs and evaluation results under it are not pre-populated.

## Licensing

The datasets are **not** covered by this repository's Apache License 2.0. Each
dataset remains under the license and terms of use of its original source, and
this repository does not grant any rights to redistribute it. Check and follow
the original source's terms before redistributing the data or the task
metadata, and cite the original datasets in academic work that uses them.
