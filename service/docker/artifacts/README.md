# Detection artifacts for GPU image

These JSON files are copied into the GPU Docker image so detection works without a volume mount.

- `normalization_098.json`
- `likelihood_params.json`
- `calibration_098_001.json`

To refresh from your results: `cp ../../results/*.json .`
