# Datasets

Place local datasets, dataset manifests, and derived feature caches here.

Recommended layout:

```text
datasets/VGGSound-Sparse/
```

Dataset contents are ignored by Git. Keep large videos, audio, feature tensors, and generated dataset artifacts out of version control.

The public demo videos are tracked intentionally:

```text
datasets/demo_videos/cat.mp4
datasets/demo_videos/dog.mp4
```

Keep all other dataset files and feature caches untracked.
