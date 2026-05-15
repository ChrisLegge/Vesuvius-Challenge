# Model Metadata

This folder tracks lightweight JSON metadata for trained models.

It is safe to commit these files because they are small and describe experiments without publishing large trained weights.

Tracked metadata includes:

- model role
- seed and fold
- patch size
- best validation proxy score
- validation metric breakdown
- training phase changes
- inference thresholds
- runtime settings

The corresponding `.pt` checkpoint files are excluded from Git.
