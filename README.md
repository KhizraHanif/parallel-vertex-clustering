# GPU Vertex Clustering for 3D Mesh Reduction

![Mesh Simplification](fig/simplification-example.png)

## About

This project extends the P-Weld algorithm by Fathollahi & Chester
(SIGGRAPH Asia 2023) with two fully GPU-resident implementations of
strict-mode vertex clustering for 3D mesh reduction.

The original P-Weld algorithm introduced a lock-free, dependency-driven
approach to parallel vertex clustering using atomic primitives. This work
ports and optimises that algorithm for NVIDIA GPUs, eliminating CPU
involvement in the clustering pipeline entirely.

![Clustering Example](fig/clustering-example.png)

## Contributions

Building on the original CPU P-Weld implementation, this project adds:

**Model 1: On-the-Fly GPU Clustering**
A direct GPU port of P-Weld's dependency-driven logic. The hash grid is
built once on the GPU and neighbor relationships are discovered on the fly
during each clustering iteration — no precomputed adjacency list.

**Model 2: GPU Streaming Clustering (main contribution)**
A fully GPU-resident pipeline that precomputes a CSR adjacency list once
using a two-pass hash grid approach, then reuses it across all clustering
iterations. Key optimisations include:
- SplitMix64 open-addressing hash table for O(1) cell lookup
- Warp-buffered frontier kernel with match_any_sync aggregation
- Hybrid dense/light neighbor search paths for load balancing
- cudaOccupancyMaxPotentialBlockSize for adaptive launch tuning

Both GPU models produce bit-identical output to the original CPU P-Weld.

Also included is a reusable standalone GPU neighbor search module
(cpp/neighbor_list.cu) that can be dropped into other CUDA projects.

## Requirements

- NVIDIA GPU (Volta or newer, sm_70+)
- CUDA Toolkit 12.x
- CMake 3.18+
- GCC compatible with your CUDA version (GCC 11 recommended for CUDA 12.x)
- Linux (tested on Ubuntu 22.04)

## Build

```bash
git clone --recurse-submodules https://github.com/KhizraHanif/parallel-vertex-clustering.git
cd parallel-vertex-clustering
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 ..
make -j$(nproc)
```

For best performance on your specific GPU, set the architecture explicitly:

```bash
# RTX 3080/3090 (Ampere)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..

# RTX 4080/4090 (Ada)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 ..

# RTX 5070 Ti/5080/5090 (Blackwell)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 ..
```

## Usage

```bash
./merge-vertices <eps> <version> <mesh.ply> <cores> [output.ply]
```

| Version | Description | Authors |
|---------|-------------|---------|
| `0` | S-Weld — Open3D baseline | Fathollahi & Chester |
| `1` | P-Weld — CPU multicore lock-free clustering | Fathollahi & Chester |
| `2` | P-Weld Async — asynchronous CPU variant | Fathollahi & Chester |
| `3` | Hybrid — CPU KDTree search + GPU frontier clustering | Hanif |
| `4` | Model 2: GPU Streaming — fully GPU, precomputed CSR | Hanif |
| `5` | Model 1: GPU On-the-Fly — fully GPU, no precomputed CSR | Hanif |

Example:
```bash
./merge-vertices 0.01 4 ../data/xyzrgb_manuscript.ply 1 output.ply
```

Note: only .ply format is supported.

## Finding the right epsilon

Use the included epsilon-finder to find an epsilon value corresponding
to a desired vertex reduction rate:

```bash
./epsilon-finder <mesh.ply> <target-reduction-rate> <threads>
```

Example — reduce to 10% of original vertices using 4 threads:
```bash
./epsilon-finder ../data/xyzrgb_manuscript.ply 0.1 4
```

Note: epsilon-finder uses the CPU P-Weld implementation and is intended
for offline calibration before running the GPU versions.

## Attribution

This project extends the P-Weld implementation by
[Fathollahi & Chester](https://github.com/nimaft97/parallel-vertex-clustering).
The CPU clustering code (cpp/TriangleMeshPWeld.cpp, cpp/KDTreeFlann.cpp,
and related files) is adapted from their original work. The GPU implementations
in cpp/PWeldCuda.cu and cpp/neighbor_list.cu are original contributions.

## Citation

If you use code cpu versions of code in your research, please cite the original P-Weld paper:

```bibtex
@inproceedings{10.1145/3610548.3618234,
  author    = {Fathollahi, Nima and Chester, Sean},
  title     = {Lock-Free Vertex Clustering for Multicore Mesh Reduction},
  year      = {2023},
  url       = {https://doi.org/10.1145/3610548.3618234},
  doi       = {10.1145/3610548.3618234},
  booktitle = {{SIGGRAPH} Asia 2023 Conference Papers},
  articleno = {60},
  numpages  = {10}
}
```
