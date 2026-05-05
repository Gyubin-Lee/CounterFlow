# Patches

This directory stores reproducible patches for external repositories.

MMAudio:

```bash
cd external/MMAudio
git apply ../../patches/mmaudio_networks_counterflow.patch
```

av-benchmark:

```bash
cd external/av-benchmark
git apply ../../patches/av_benchmark_counterflow.patch
```

Apply only the patches needed for the experiment you are reproducing.
