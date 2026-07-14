// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2025 Khizra Hanif
//
// GPU implementation of strict-mode vertex clustering for 3D mesh reduction.
// Extends the P-Weld algorithm by Fathollahi & Chester (SIGGRAPH Asia 2023)
// with two fully GPU-resident implementations:
//   Model 1: On-the-Fly GPU clustering (no precomputed adjacency)
//   Model 2: GPU Streaming clustering with precomputed CSR adjacency
// ----------------------------------------------------------------------------

#include <cfloat>
#include <cuda_runtime.h>
#include <cuNSearch.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <Eigen/Dense>
#include <cub/cub.cuh>
#include <thrust/iterator/constant_iterator.h>

#include <omp.h>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CUDA_RESTRICT __restrict__

bool enable_injection = false;

#define MAX_CELL_SIZE 64
#define WARP_SIZE 32
#define SHARED_FRONTIER_CAP 512

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Hash a 3D cell index into a 64-bit key using large prime multipliers.
__host__ __device__ inline long long cell_key(int ix, int iy, int iz) {
    const long long p1 = 73856093;
    const long long p2 = 19349663;
    const long long p3 = 83492791;
    return ((long long)ix * p1) ^ ((long long)iy * p2) ^ ((long long)iz * p3);
}

// Structure-of-Arrays layout for vertex positions on device.
struct PointsSOA {
    double *x, *y, *z;
};

// Spatial grid metadata.
struct GridInfo {
    double3 min_bound;
    double cell_size;
    int3 resolution;
};

// 64-bit cell key type alias for clarity.
using CellKeyT = long long;

// ============================================================
// GPU Timer
// ============================================================
inline float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

class GpuTimer {
public:
    GpuTimer(const std::string& label, bool print = true, cudaStream_t stream = 0)
        : label_(label), print_(print), elapsed_ms_(0.0f), stream_(stream)
    {
        cudaEventCreateWithFlags(&start_, cudaEventDefault);
        cudaEventCreateWithFlags(&stop_,  cudaEventDefault);
        cudaEventRecord(start_, stream_);
        if (print_) indent_ = depth_++;
    }

    double seconds() {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_ms_, start_, stop_);
        return static_cast<double>(elapsed_ms_) / 1000.0;
    }

    ~GpuTimer() {
        if (print_) {
            cudaEventRecord(stop_, stream_);
            cudaEventSynchronize(stop_);
            cudaEventElapsedTime(&elapsed_ms_, start_, stop_);
            for (int i = 0; i < indent_; i++) std::cout << "    ";
            std::cout << label_ << " took " << elapsed_ms_ / 1000.0 << " seconds\n";
            depth_--;
        }
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    cudaStream_t stream_ = 0;
    std::string  label_;
    bool         print_;
    cudaEvent_t  start_, stop_;
    float        elapsed_ms_;
    int          indent_ = 0;
    static thread_local int depth_;
};
thread_local int GpuTimer::depth_ = 0;


namespace open3d {
namespace geometry {

// ============================================================
// PCIe Injection Utilities (Amdahl's Law experiments)
// ============================================================
static void* gpuinj_h_pinned = nullptr;
static void* gpuinj_d_buf    = nullptr;
static size_t gpuinj_bytes   = 0;

static void gpuinj_init_buffers(size_t bytes) {
    if (bytes == 0) return;
    gpuinj_bytes = bytes;
    cudaMallocHost(&gpuinj_h_pinned, bytes);
    cudaMalloc(&gpuinj_d_buf, bytes);
    memset(gpuinj_h_pinned, 0x3f, bytes);
    cudaMemset(gpuinj_d_buf, 0x5a, bytes);
}

static void gpuinj_free_buffers() {
    if (gpuinj_d_buf)    cudaFree(gpuinj_d_buf);
    if (gpuinj_h_pinned) cudaFreeHost(gpuinj_h_pinned);
    gpuinj_d_buf    = nullptr;
    gpuinj_h_pinned = nullptr;
    gpuinj_bytes    = 0;
}

static double gpuinj_do_copies(int copiesPerIter, cudaStream_t stream) {
    if (copiesPerIter <= 0 || gpuinj_bytes == 0) return 0.0;
    cudaEvent_t eStart, eStop;
    cudaEventCreate(&eStart); cudaEventCreate(&eStop);
    cudaEventRecord(eStart, stream);
    for (int k = 0; k < copiesPerIter; ++k) {
        cudaMemcpyAsync(gpuinj_d_buf, gpuinj_h_pinned, gpuinj_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(gpuinj_h_pinned, gpuinj_d_buf, gpuinj_bytes, cudaMemcpyDeviceToHost, stream);
    }
    cudaEventRecord(eStop, stream);
    cudaEventSynchronize(eStop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, eStart, eStop);
    cudaEventDestroy(eStart);
    cudaEventDestroy(eStop);
    return ms * 1e-3f;
}

// ============================================================
// Kernel: Baseline strict frontier clustering (Version 3)
// Uses precomputed CSR adjacency from CPU KDTree.
// ============================================================
__global__ void strict_frontier_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    int* __restrict__ next_frontier,
    int* __restrict__ next_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= active_count) return;

    int u = active_vertices[tid];
    if (u < 0 || u >= nverts) return;

    depend[u] = -1;
    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    int start = neighbor_offsets[u];
    int end   = neighbor_offsets[u + 1];

    for (int i = start; i < end; i++) {
        int v = neighbor_indices[i];
        if (v < 0 || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        int old = atomicAdd(&depend[v], -1);
        if (old == 1) {
            int pos = atomicAdd(next_count, 1);
            if (pos < nverts) next_frontier[pos] = v;
        }
    }
}

// ============================================================
// Hash table: SplitMix64-style mixing for open-addressing
// ============================================================
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Build GPU open-addressing hash table from sorted cell keys.
__global__ void build_hash_table_kernel(
    const long long* __restrict__ cell_keys,
    int num_cells,
    long long* __restrict__ hash_keys,
    int* __restrict__ hash_vals,
    long long table_mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_cells) return;

    unsigned long long h    = mix64((unsigned long long)cell_keys[tid]);
    unsigned int       slot = h & table_mask;
    unsigned int       step = ((h >> 32) | 1u);

    while (true) {
        unsigned long long prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
            (unsigned long long)LLONG_MIN,
            (unsigned long long)cell_keys[tid]);

        if (prev == (unsigned long long)LLONG_MIN ||
            prev == (unsigned long long)cell_keys[tid]) {
            hash_vals[slot] = tid;
            return;
        }
        slot = (slot + step) & table_mask;
    }
}

// O(1) expected-case cell lookup in open-addressing hash table.
__device__ __forceinline__ int hash_lookup(
    long long key,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    long long table_mask)
{
    unsigned long long h    = mix64((unsigned long long)key);
    unsigned int       slot = h & table_mask;
    unsigned int       step = ((h >> 32) | 1u);

    while (true) {
        long long stored = hash_keys[slot];
        if (stored == key)      return hash_vals[slot];
        if (stored == LLONG_MIN) return -1;
        slot = (slot + step) & table_mask;
    }
}

// Binary upper-bound search within a sorted cell segment.
__device__ __forceinline__ int upper_bound_int(const int* a, int len, int val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&a[mid]) <= val) lo = mid + 1;
        else                        hi = mid;
    }
    return lo;
}

// ============================================================
// Kernel: Count neighbors per vertex using compact hash grid.
// Hybrid dense/light path for load balancing across cell sizes.
// ============================================================
__global__ void count_neighbors_compact_kernel(
    const double3* __restrict__ points,
    int* __restrict__ neighbor_counts,
    int nverts, double eps, double3 min_bound, double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    const int* __restrict__ occupied_cells,
    int num_occupied,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    extern __shared__ int sh_points[];
    const int    TILE = 256;
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < nverts; u += blockDim.x * gridDim.x) {
        const double3 pu    = points[u];
        int           count = 0;

        const int cx = __double2int_rd((pu.x - min_bound.x) / cell_size);
        const int cy = __double2int_rd((pu.y - min_bound.y) / cell_size);
        const int cz = __double2int_rd((pu.z - min_bound.z) / cell_size);

#pragma unroll
        for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const long long key = cell_key(cx + dx, cy + dy, cz + dz);
            const int       idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
            if (idx < 0) continue;

            const int beg = __ldg(&cell_offsets[idx]);
            const int end = __ldg(&cell_offsets[idx + 1]);
            const int len = end - beg;
            if (len <= 0) continue;

            int pos = beg + upper_bound_int(cell_points + beg, len, u);

            if (len > TILE) {
                // Dense-cell path: cooperative shared-memory tiling
                for (int t = pos; t < end; t += TILE) {
                    const int chunk = min(TILE, end - t);
                    for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                        sh_points[i] = __ldg(&cell_points[t + i]);
                    __syncthreads();
#pragma unroll 4
                    for (int i = 0; i < chunk; ++i) {
                        const int    v   = sh_points[i];
                        const double3 pv = points[v];
                        const double dx2 = pv.x - pu.x, dy2 = pv.y - pu.y, dz2 = pv.z - pu.z;
                        count += (dx2*dx2 + dy2*dy2 + dz2*dz2 <= eps2);
                    }
                    __syncthreads();
                }
            } else {
                // Light-cell path: direct global memory reads
#pragma unroll 4
                for (int k = pos; k < end; ++k) {
                    const int    v   = __ldg(&cell_points[k]);
                    const double3 pv = points[v];
                    const double dx2 = pv.x - pu.x, dy2 = pv.y - pu.y, dz2 = pv.z - pu.z;
                    count += (dx2*dx2 + dy2*dy2 + dz2*dz2 <= eps2);
                }
            }
        }
        neighbor_counts[u] = count;
    }
}

// ============================================================
// Kernel: Initialize depend[] from prebuilt CSR adjacency.
// ============================================================
__global__ void init_depend_from_csr_kernel(
    int* __restrict__ depend,
    int nverts,
    const int* __restrict__ neighbor_offsets,
    const int* __restrict__ neighbor_indices)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nverts) return;

    int start = neighbor_offsets[u];
    int end   = neighbor_offsets[u + 1];

    for (int i = start; i < end; ++i) {
        int v = neighbor_indices[i];
        if (v > u) atomicAdd(&depend[v], 1);
    }
}

// ============================================================
// Kernel: On-the-fly streaming clustering iteration (Model 1).
// Neighbor search is performed every iteration via hash grid.
// ============================================================
__global__ __launch_bounds__(256, 2)
void streaming_iteration_hash_kernel(
    const double3* points,
    int* cp_vec,
    int* depend,
    bool* changed_flags,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    const long long* cell_keys,
    const int* cell_offsets,
    const int* cell_points,
    int nkeys,
    const long long* hash_keys,
    const int* hash_vals,
    long long table_mask)
{
    int  tid          = blockIdx.x * blockDim.x + threadIdx.x;
    bool local_changed = false;

    for (int u = tid; u < nverts; u += gridDim.x * blockDim.x) {
        if (depend[u] != 0) continue;

        depend[u]      = -1;
        bool isCentroid = (cp_vec[u] == u);
        double3 pu      = points[u];

        int ix0 = (int)((pu.x - min_bound.x) / cell_size);
        int iy0 = (int)((pu.y - min_bound.y) / cell_size);
        int iz0 = (int)((pu.z - min_bound.z) / cell_size);

#pragma unroll
        for (int dx = -1; dx <= 1; dx++)
#pragma unroll
        for (int dy = -1; dy <= 1; dy++)
#pragma unroll
        for (int dz = -1; dz <= 1; dz++) {
            long long key = cell_key(ix0 + dx, iy0 + dy, iz0 + dz);
            int       idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
            if (idx < 0) continue;

            int start = cell_offsets[idx];
            int end   = cell_offsets[idx + 1];

            for (int i = start; i < end; i++) {
                int v = cell_points[i];
                if (u == v) continue;

                double dx_ = pu.x - points[v].x;
                double dy_ = pu.y - points[v].y;
                double dz_ = pu.z - points[v].z;

                if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps * eps) {
                    if (isCentroid && depend[v] > 0) {
                        int expected = cp_vec[v];
                        int desired  = u;
                        while (desired < expected) {
                            int old = atomicCAS(&cp_vec[v], expected, desired);
                            if (old == expected) break;
                            expected = old;
                        }
                    }
                    int old = atomicSub(&depend[v], 1);
                    if (old == 1) local_changed = true;
                }
            }
        }
    }

    if (tid < nverts) changed_flags[tid] = local_changed;
}

// ============================================================
// Kernel: Fill CSR neighbor indices (Model 2, pass 2).
// Hybrid dense/light path matches the count-pass logic.
// ============================================================
__global__ void build_neighbors_compact_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ neighbor_offsets,
    int* __restrict__ neighbor_counts,
    int* __restrict__ neighbor_indices,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    const int* __restrict__ occupied_cells,
    int num_occupied,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    extern __shared__ int sh_points[];
    const int    TILE = 256;
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x;
         u < nverts;
         u += blockDim.x * gridDim.x)
    {
        const double3 pu       = points[u];
        const int     rowStart = neighbor_offsets[u];
        const int     rowEnd   = neighbor_offsets[u + 1];
        int           out      = rowStart;

        const int3 c0 = make_int3(
            __double2int_rd((pu.x - min_bound.x) / cell_size),
            __double2int_rd((pu.y - min_bound.y) / cell_size),
            __double2int_rd((pu.z - min_bound.z) / cell_size));

#pragma unroll
        for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const long long key = cell_key(c0.x + dx, c0.y + dy, c0.z + dz);
            const int       idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
            if (idx < 0) continue;

            const int beg = __ldg(&cell_offsets[idx]);
            const int end = __ldg(&cell_offsets[idx + 1]);
            const int len = end - beg;
            if (len <= 0) continue;

            const int pos = beg + upper_bound_int(cell_points + beg, len, u);

            if (len > TILE) {
                for (int t = pos; t < end; t += TILE) {
                    const int chunk = min(TILE, end - t);
                    for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                        sh_points[i] = __ldg(&cell_points[t + i]);
                    __syncthreads();
#pragma unroll 4
                    for (int i = 0; i < chunk; ++i) {
                        const int    v  = sh_points[i];
                        if (v <= u) continue;
                        const double3 pv = points[v];
                        const double dxf = pv.x - pu.x, dyf = pv.y - pu.y, dzf = pv.z - pu.z;
                        if (dxf*dxf + dyf*dyf + dzf*dzf <= eps2)
                            if (out < rowEnd) neighbor_indices[out++] = v;
                    }
                    __syncthreads();
                }
            } else {
#pragma unroll 4
                for (int k = pos; k < end; ++k) {
                    const int    v  = __ldg(&cell_points[k]);
                    if (v <= u) continue;
                    const double3 pv = points[v];
                    const double dxf = pv.x - pu.x, dyf = pv.y - pu.y, dzf = pv.z - pu.z;
                    if (dxf*dxf + dyf*dyf + dzf*dzf <= eps2)
                        if (out < rowEnd) neighbor_indices[out++] = v;
                }
            }
        }
    }
}

// ============================================================
// Build CSR neighbor list from prebuilt hash grid (two-pass).
// Used by Model 2 (GPU Streaming).
// ============================================================
void build_neighbor_list_from_hashgrid(
    thrust::device_vector<double3>& d_in_vertices,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    thrust::device_vector<long long>& d_cell_keys,
    thrust::device_vector<int>& d_cell_offsets,
    thrust::device_vector<int>& d_cell_points,
    thrust::device_vector<long long>& d_hash_keys,
    thrust::device_vector<int>& d_hash_vals,
    long long table_mask,
    thrust::device_vector<int>& d_neighbor_offsets,
    thrust::device_vector<int>& d_neighbor_indices,
    thrust::device_vector<int>& d_occupied_indices,
    int num_occupied,
    cudaStream_t stream = 0)
{
    GpuTimer t("Build neighbor list (two-pass CSR)", true, stream);
    auto pol = thrust::cuda::par.on(stream);

    thrust::device_vector<int> d_neighbor_counts(nverts, 0);

    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       count_neighbors_compact_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    // Pass 1: count neighbors
    {
        GpuTimer t1("Count neighbors", true, stream);
        size_t shmem = blockSize * sizeof(int);
        count_neighbors_compact_kernel<<<gridSize, blockSize, shmem, stream>>>(
            thrust::raw_pointer_cast(d_in_vertices.data()),
            thrust::raw_pointer_cast(d_neighbor_counts.data()),
            nverts, eps, min_bound, cell_size,
            thrust::raw_pointer_cast(d_cell_keys.data()),
            thrust::raw_pointer_cast(d_cell_offsets.data()),
            thrust::raw_pointer_cast(d_cell_points.data()),
            thrust::raw_pointer_cast(d_occupied_indices.data()),
            num_occupied,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()),
            table_mask);
    }

    // Prefix-sum to get CSR offsets
    d_neighbor_offsets.resize(nverts + 1);
    thrust::exclusive_scan(pol,
        d_neighbor_counts.begin(), d_neighbor_counts.end(),
        d_neighbor_offsets.begin());

    int last_count = 0, last_offset = 0;
    CUDA_CHECK(cudaMemcpyAsync(&last_count,
        thrust::raw_pointer_cast(d_neighbor_counts.data() + nverts - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_offset,
        thrust::raw_pointer_cast(d_neighbor_offsets.data() + nverts - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int total_neighbors = last_offset + last_count;
    d_neighbor_indices.resize(total_neighbors);

    thrust::fill(pol, d_neighbor_counts.begin(), d_neighbor_counts.end(), 0);

    // Pass 2: fill neighbor indices
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       build_neighbors_compact_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    {
        GpuTimer t2("Fill neighbors", true);
        size_t shmem = blockSize * sizeof(int);
        build_neighbors_compact_kernel<<<gridSize, blockSize, shmem, stream>>>(
            thrust::raw_pointer_cast(d_in_vertices.data()),
            thrust::raw_pointer_cast(d_neighbor_offsets.data()),
            thrust::raw_pointer_cast(d_neighbor_counts.data()),
            thrust::raw_pointer_cast(d_neighbor_indices.data()),
            nverts, eps, min_bound, cell_size,
            thrust::raw_pointer_cast(d_cell_keys.data()),
            thrust::raw_pointer_cast(d_cell_offsets.data()),
            thrust::raw_pointer_cast(d_cell_points.data()),
            thrust::raw_pointer_cast(d_occupied_indices.data()),
            num_occupied,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()),
            table_mask);
    }

    std::cout << "CSR neighbor list built: " << total_neighbors << " edges ("
              << std::fixed << std::setprecision(3)
              << (total_neighbors / (double)nverts)
              << " avg/vertex) from " << num_occupied << " occupied cells.\n";
}

// ============================================================
// Centroid reduction helpers
// ============================================================
struct SumCount {
    double x, y, z;
    int count;
};

struct SumCombine {
    __device__ SumCount operator()(const SumCount& a, const SumCount& b) const {
        return SumCount{ a.x+b.x, a.y+b.y, a.z+b.z, a.count+b.count };
    }
};

// ============================================================
// Kernel: Remap face indices + flag degenerate triangles.
// ============================================================
__global__ void remap_faces_kernel_flags(
    const int3* __restrict__ in_faces,
    int3* __restrict__ out_faces,
    int* __restrict__ flags,
    const int* __restrict__ pid2ccid,
    int nfaces)
{
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= nfaces) return;

    int3 f = in_faces[fid];
    int a = pid2ccid[f.x], b = pid2ccid[f.y], c = pid2ccid[f.z];

    if (a == b || b == c || a == c) { flags[fid] = 0; return; }
    out_faces[fid] = make_int3(a, b, c);
    flags[fid] = 1;
}

// ============================================================
// Kernel: Warp-buffered frontier clustering (Model 2).
// One warp per active vertex; shared-memory frontier buffer
// reduces global atomic contention.
// ============================================================
__global__ void strict_frontier_warp_blockBuffered_kernel(
    const PointsSOA points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    int* __restrict__ next_frontier,
    int* __restrict__ next_count)
{
    extern __shared__ int shared_frontier[];
    __shared__ int local_count;
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id    = global_tid / WARP_SIZE;
    int lane_id    = threadIdx.x & 31;

    if (warp_id >= active_count) return;

    int u = active_vertices[warp_id];
    if (u < 0 || u >= nverts) return;

    depend[u]       = -1;
    int  cu         = cp_vec[u];
    bool isCentroid = (cu == u);

    int start  = neighbor_offsets[u];
    int end    = neighbor_offsets[u + 1];
    int degree = end - start;

    for (int i = lane_id; i < degree; i += WARP_SIZE) {
        int v = neighbor_indices[start + i];
        if (v <= u || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        unsigned mask       = __match_any_sync(__activemask(), v);
        int      leader     = __ffs(mask) - 1;
        int      group_size = __popc(mask);

        int old = 0;
        if (lane_id == leader) old = atomicAdd(&depend[v], -group_size);
        old = __shfl_sync(mask, old, leader);

        bool became_zero = (old <= group_size && old > 0);
        if (became_zero) {
            int pos = atomicAdd(&local_count, 1);
            if (pos < SHARED_FRONTIER_CAP) {
                shared_frontier[pos] = v;
            } else {
                if (lane_id == leader) {
                    int idx = atomicAdd(next_count, 1);
                    if (idx < nverts) next_frontier[idx] = v;
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        int idx = atomicAdd(next_count, 1);
        if (idx < nverts) next_frontier[idx] = shared_frontier[i];
    }
}

// ============================================================
// Kernel: face remap + valid face count (single pass).
// ============================================================
__global__ void remap_and_count_kernel(
    const int3* __restrict__ in_faces,
    int3* __restrict__ out_faces,
    const int* __restrict__ pid2ccid,
    int nfaces,
    unsigned int* __restrict__ valid_counter)
{
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= nfaces) return;

    int3 f = in_faces[fid];
    int a = pid2ccid[f.x], b = pid2ccid[f.y], c = pid2ccid[f.z];

    if (a == b || b == c || a == c) {
        out_faces[fid] = make_int3(-1, -1, -1);
        return;
    }
    out_faces[fid] = make_int3(a, b, c);
    atomicAdd(valid_counter, 1u);
}

// ============================================================
// Version 3: Hybrid GPU clustering (CPU KDTree + GPU frontier)
// ============================================================
void merge_vertices_forward_gpu(
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<Eigen::Vector3i>& triangles,
    const std::vector<int>& neighbor_indices,
    const std::vector<int>& neighbor_offsets,
    const std::vector<int>& depend_init_host,
    double eps,
    bool print_time)
{
    GpuTimer total("timeAll_s", print_time);

    int nverts = (int)vertices.size();
    int nfaces = (int)triangles.size();

    {
        GpuTimer t("timeP_s", print_time);
        if (!depend_init_host.empty()) {
            bool has_nonzero = std::any_of(depend_init_host.begin(), depend_init_host.end(),
                                           [](int x){ return x > 0; });
            if (has_nonzero)
                std::cout << "Reusing depend[] from KDTree adjacency (strict-mode)\n";
        }
    }

    thrust::device_vector<double3> d_in_vertices;
    thrust::device_vector<int> d_cp_vec, d_depend, d_neighbors, d_offsets, d_pid2ccid;
    {
        std::vector<double3> h_in_vertices(nverts);
        for (int i = 0; i < nverts; i++)
            h_in_vertices[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());

        d_in_vertices = h_in_vertices;
        d_cp_vec.resize(nverts);
        thrust::sequence(d_cp_vec.begin(), d_cp_vec.end());
        d_depend    = depend_init_host;
        d_neighbors = neighbor_indices;
        d_offsets   = neighbor_offsets;
    }

    {
        GpuTimer clustering("Step 2: Baseline Frontier Clustering", print_time);
        const int BLOCK_SIZE = 256;
        thrust::device_vector<int> d_active(nverts), d_next_active(nverts), d_next_count(1);

        auto end_it = thrust::copy_if(
            thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nverts),
            d_depend.begin(), d_active.begin(),
            [] __device__(int dep){ return dep == 0; });
        int active_count = end_it - d_active.begin();

        int iter = 0;
        while (active_count > 0) {
            thrust::fill(d_next_count.begin(), d_next_count.end(), 0);
            int numBlocks = (active_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            strict_frontier_kernel<<<numBlocks, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(d_in_vertices.data()),
                thrust::raw_pointer_cast(d_cp_vec.data()),
                thrust::raw_pointer_cast(d_depend.data()),
                thrust::raw_pointer_cast(d_active.data()),
                active_count,
                thrust::raw_pointer_cast(d_neighbors.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                nverts,
                thrust::raw_pointer_cast(d_next_active.data()),
                thrust::raw_pointer_cast(d_next_count.data()));
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_next_count = 0;
            cudaMemcpy(&h_next_count,
                       thrust::raw_pointer_cast(d_next_count.data()),
                       sizeof(int), cudaMemcpyDeviceToHost);
            if (h_next_count > 0)
                thrust::copy_n(d_next_active.begin(), h_next_count, d_active.begin());
            active_count = h_next_count;
            iter++;
        }
        std::cout << "        (iters = " << iter << ")\n";
    }

    {
        GpuTimer tSingle("timeSingle_s", print_time);
        thrust::device_vector<int> d_is_centroid(nverts), d_prefix(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(nverts),
            d_is_centroid.begin(),
            [cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data())] __device__(int i){
                return (cp_ptr[i] == i) ? 1 : 0; });
        thrust::exclusive_scan(d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());

        d_pid2ccid.resize(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(nverts),
            d_pid2ccid.begin(),
            [cp_ptr   = thrust::raw_pointer_cast(d_cp_vec.data()),
             pfx_ptr  = thrust::raw_pointer_cast(d_prefix.data())] __device__(int i){
                return pfx_ptr[cp_ptr[i]]; });

        int nclusters, last_flag;
        cudaMemcpy(&nclusters,  thrust::raw_pointer_cast(d_prefix.data()     + nverts - 1), sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_flag,  thrust::raw_pointer_cast(d_is_centroid.data() + nverts - 1), sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "        (clusters = " << nclusters + last_flag << ")\n";
    }

    int reduced_clusters = 0;
    thrust::device_vector<int> d_unique_keys, d_keys;
    thrust::device_vector<SumCount> d_vals, d_sums;

    {
        GpuTimer tUR("timeUR_s", print_time);
        d_keys.resize(nverts); d_vals.resize(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(nverts),
            d_keys.begin(),
            [p = thrust::raw_pointer_cast(d_pid2ccid.data())] __device__(int i){ return p[i]; });
        thrust::transform(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(nverts),
            d_vals.begin(),
            [vp = thrust::raw_pointer_cast(d_in_vertices.data())] __device__(int i){
                double3 v = vp[i]; return SumCount{v.x, v.y, v.z, 1}; });
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());
        d_unique_keys.resize(nverts); d_sums.resize(nverts);
        auto new_end = thrust::reduce_by_key(
            d_keys.begin(), d_keys.end(), d_vals.begin(),
            d_unique_keys.begin(), d_sums.begin(),
            thrust::equal_to<int>(), SumCombine());
        reduced_clusters = new_end.first - d_unique_keys.begin();
    }

    {
        GpuTimer tUNV("timeUNV_s", print_time);
        thrust::device_vector<double3> d_out_vertices(reduced_clusters);
        thrust::transform(
            d_sums.begin(), d_sums.begin() + reduced_clusters,
            d_out_vertices.begin(),
            [] __device__(const SumCount& sc){
                return make_double3(sc.x/sc.count, sc.y/sc.count, sc.z/sc.count); });
        d_in_vertices.swap(d_out_vertices);
        std::cout << "        (reduced clusters = " << reduced_clusters << ")\n";
    }

    int valid_faces = 0;
    thrust::device_vector<int3> d_compact_faces;
    {
        GpuTimer tU("timeU_s", print_time);
        std::vector<int3> h_in_faces(nfaces);
        for (int i = 0; i < nfaces; i++)
            h_in_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
        thrust::device_vector<int3> d_in_faces = h_in_faces;
        thrust::device_vector<int3> d_out_faces(nfaces);
        thrust::device_vector<int>  d_flags(nfaces);

        remap_faces_kernel_flags<<<(nfaces+255)/256, 256>>>(
            thrust::raw_pointer_cast(d_in_faces.data()),
            thrust::raw_pointer_cast(d_out_faces.data()),
            thrust::raw_pointer_cast(d_flags.data()),
            thrust::raw_pointer_cast(d_pid2ccid.data()),
            nfaces);
        CUDA_CHECK(cudaDeviceSynchronize());

        d_compact_faces.resize(nfaces);
        auto compact_end = thrust::copy_if(
            d_out_faces.begin(), d_out_faces.end(),
            d_flags.begin(), d_compact_faces.begin(),
            [] __device__(int flag){ return flag == 1; });
        valid_faces = compact_end - d_compact_faces.begin();
        std::cout << "    (valid faces = " << valid_faces << ")\n";
    }

    {
        thrust::host_vector<double3> h_out = d_in_vertices;
        vertices.clear(); vertices.reserve(h_out.size());
        for (auto& v : h_out) vertices.emplace_back(v.x, v.y, v.z);

        thrust::host_vector<int3> h_faces(d_compact_faces.begin(),
                                           d_compact_faces.begin() + valid_faces);
        triangles.clear(); triangles.reserve(valid_faces);
        for (auto& f : h_faces) triangles.emplace_back(f.x, f.y, f.z);
    }

    std::cout << "Final compact clusters: " << vertices.size()
              << " | Final faces: " << triangles.size() << "\n";
}

// ============================================================
// Kernel: compute per-vertex cell hash keys on device.
// ============================================================
__global__ void compute_keys_kernel(
    const double3* pts, long long* keys, int n,
    double3 minb, double cell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double3 p  = pts[i];
    int     ix = (int)((p.x - minb.x) / cell);
    int     iy = (int)((p.y - minb.y) / cell);
    int     iz = (int)((p.z - minb.z) / cell);
    keys[i] = ((long long)ix * 73856093LL)
            ^ ((long long)iy * 19349663LL)
            ^ ((long long)iz * 83492791LL);
}

// ============================================================
// Build sparse hash grid entirely on GPU.
// ============================================================
void build_hash_grid_gpu_safe(
    const std::vector<Eigen::Vector3d>& vertices,
    double eps,
    GridInfo& grid,
    thrust::device_vector<long long>& d_cell_keys,
    thrust::device_vector<int>& d_cell_offsets,
    thrust::device_vector<int>& d_cell_points,
    bool print_time = true,
    cudaStream_t stream = 0)
{
    int nverts = static_cast<int>(vertices.size());
    if (nverts == 0) return;

    auto pol = thrust::cuda::par.on(stream);

    Eigen::Vector3d min_bound(DBL_MAX, DBL_MAX, DBL_MAX);
    Eigen::Vector3d max_bound(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (auto& v : vertices) {
        min_bound = min_bound.cwiseMin(v);
        max_bound = max_bound.cwiseMax(v);
    }

    grid.min_bound    = make_double3(min_bound.x(), min_bound.y(), min_bound.z());
    grid.cell_size    = eps;
    grid.resolution.x = static_cast<int>((max_bound.x() - min_bound.x()) / eps) + 3;
    grid.resolution.y = static_cast<int>((max_bound.y() - min_bound.y()) / eps) + 3;
    grid.resolution.z = static_cast<int>((max_bound.z() - min_bound.z()) / eps) + 3;

    std::cout << "Grid resolution = ("
              << grid.resolution.x << ", "
              << grid.resolution.y << ", "
              << grid.resolution.z << ")\n";

    long long total_cells = 1LL * grid.resolution.x * grid.resolution.y * grid.resolution.z;
    if (total_cells > INT_MAX)
        std::cout << "Warning: large grid (" << total_cells
                  << " cells), using 64-bit keys.\n";

    double3* h_points = nullptr;
    cudaMallocHost(&h_points, nverts * sizeof(double3));

    thrust::device_vector<double3> d_points(nverts);
    for (int i = 0; i < nverts; ++i)
        h_points[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_points.data()),
                    h_points, nverts * sizeof(double3),
                    cudaMemcpyHostToDevice, stream);

    thrust::device_vector<CellKeyT> d_keys(nverts);
    thrust::device_vector<int>      d_idx(nverts);
    thrust::sequence(d_idx.begin(), d_idx.end());

    const int BLOCK = 256;
    const int GRID  = (nverts + BLOCK - 1) / BLOCK;
    compute_keys_kernel<<<GRID, BLOCK, 0, stream>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        nverts, grid.min_bound, grid.cell_size);

    {
        GpuTimer t("GPU sort_by_key (hash grid)", print_time, stream);
        if (d_keys.size() > 0) {
            thrust::sort_by_key(thrust::device,
                                d_keys.begin(), d_keys.end(), d_idx.begin());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    thrust::device_vector<CellKeyT> d_unique_keys(nverts);
    thrust::device_vector<int>      d_offsets(nverts + 1);
    thrust::device_vector<int>      d_counts(nverts);

    auto end_pair = thrust::reduce_by_key(pol,
        d_keys.begin(), d_keys.end(),
        thrust::make_constant_iterator(1),
        d_unique_keys.begin(), d_counts.begin());

    int num_cells = end_pair.first - d_unique_keys.begin();
    d_unique_keys.resize(num_cells);
    d_counts.resize(num_cells);

    thrust::exclusive_scan(pol, d_counts.begin(), d_counts.end(), d_offsets.begin());
    d_offsets[num_cells] = nverts;

    d_cell_keys.swap(d_unique_keys);
    d_cell_offsets.swap(d_offsets);
    d_cell_points.swap(d_idx);

    std::cout << "GPU hash grid built: " << num_cells
              << " occupied cells, " << nverts << " points.\n";

    cudaFreeHost(h_points);
}

// ============================================================
// Model 2: GPU Streaming Strict-Mode Vertex Clustering
// Fully GPU-resident: hash grid + CSR adjacency + frontier loop
// ============================================================
void merge_vertices_forward_gpu_streaming(
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<Eigen::Vector3i>& triangles,
    double eps,
    bool enable_injection,
    bool print_time)
{
    int nverts = static_cast<int>(vertices.size());
    int nfaces = static_cast<int>(triangles.size());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto pol   = thrust::cuda::par.on(stream);
    auto first = thrust::make_counting_iterator<int>(0);
    auto last  = first + nverts;

    size_t injectMB = 64, injectCopies = 1;
    if (enable_injection)
        gpuinj_init_buffers(injectMB * 1024ULL * 1024ULL);

    double timeGrid_s = 0, timeHash_s = 0, timeCSR_s = 0, timeP_s = 0, timeSingle_s = 0;
    double timeWhile_s = 0, timeUmesh_s = 0;
    double pcie_h2d_s = 0, pcie_d2h_s = 0, total_pcie_seconds = 0;

    // Step 0: Build GPU hash grid
    GridInfo grid;
    thrust::device_vector<long long> d_cell_keys;
    thrust::device_vector<int> d_cell_offsets, d_cell_points;
    {
        GpuTimer t("Build hash grid", print_time, stream);
        build_hash_grid_gpu_safe(vertices, eps, grid,
                                 d_cell_keys, d_cell_offsets, d_cell_points,
                                 print_time, stream);
        timeGrid_s = t.seconds();
    }

    // Step 1: Build GPU hash table
    int num_cells = static_cast<int>(d_cell_keys.size());
    int table_size = 1;
    while (table_size < num_cells * 4) table_size <<= 1;
    long long table_mask = table_size - 1;

    thrust::device_vector<long long> d_hash_keys(table_size, LLONG_MIN);
    thrust::device_vector<int>       d_hash_vals(table_size, -1);
    {
        GpuTimer t("Build hash table", print_time, stream);
        int threads = 256, blocks = (num_cells + threads - 1) / threads;
        build_hash_table_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_cell_keys.data()), num_cells,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()), table_mask);
        timeHash_s = t.seconds();
    }
    std::cout << "GPU hash table built: " << table_size
              << " slots for " << num_cells << " cells.\n";

    // Step 2: Compact occupied cells
    thrust::device_vector<int> d_occupied_indices(num_cells);
    const int* offsets = thrust::raw_pointer_cast(d_cell_offsets.data());
    auto end_it = thrust::copy_if(pol, first, first + num_cells,
        d_occupied_indices.begin(),
        [offsets] __device__(int i){ return (offsets[i+1] - offsets[i]) > 0; });
    int num_occupied = static_cast<int>(end_it - d_occupied_indices.begin());
    d_occupied_indices.resize(num_occupied);
    std::cout << "Occupied cells: " << num_occupied << " / " << num_cells << "\n";

    // Step 3: Transfer vertices H->D (SoA layout)
    PointsSOA d_points, h_points;
    {
        GpuTimer t("PCIe H->D transfer", print_time, stream);
        cudaMalloc(&d_points.x, nverts * sizeof(double));
        cudaMalloc(&d_points.y, nverts * sizeof(double));
        cudaMalloc(&d_points.z, nverts * sizeof(double));
        cudaMallocHost(&h_points.x, nverts * sizeof(double));
        cudaMallocHost(&h_points.y, nverts * sizeof(double));
        cudaMallocHost(&h_points.z, nverts * sizeof(double));
        for (int i = 0; i < nverts; ++i) {
            h_points.x[i] = vertices[i].x();
            h_points.y[i] = vertices[i].y();
            h_points.z[i] = vertices[i].z();
        }
        cudaMemcpyAsync(d_points.x, h_points.x, nverts*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_points.y, h_points.y, nverts*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_points.z, h_points.z, nverts*sizeof(double), cudaMemcpyHostToDevice, stream);
        pcie_h2d_s = t.seconds();
    }

    const double* px = d_points.x, *py = d_points.y, *pz = d_points.z;

    // Step 4: Build CSR neighbor list
    thrust::device_vector<int> d_neighbor_offsets(nverts + 1);
    thrust::device_vector<int> d_neighbor_indices;
    thrust::device_vector<int3> d_in_faces(nfaces), d_out_faces(nfaces);
    thrust::device_vector<int>  d_flags(nfaces);

    {
        GpuTimer t("Build neighbor list (CSR)", print_time, stream);
        thrust::device_vector<double3> d_tmp(nverts);
        thrust::transform(pol, first, last, d_tmp.begin(),
            [px, py, pz] __device__(int i){ return make_double3(px[i], py[i], pz[i]); });
        build_neighbor_list_from_hashgrid(
            d_tmp, nverts, eps, grid.min_bound, grid.cell_size,
            d_cell_keys, d_cell_offsets, d_cell_points,
            d_hash_keys, d_hash_vals, table_mask,
            d_neighbor_offsets, d_neighbor_indices,
            d_occupied_indices, num_occupied, stream);
        timeCSR_s = t.seconds();
    }

    // Step 5: Init depend[] from CSR
    thrust::device_vector<int> d_depend(nverts, 0);
    {
        GpuTimer t("Init depend from CSR", print_time, stream);
        int threads = 256, blocks = (nverts + threads - 1) / threads;
        init_depend_from_csr_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_depend.data()), nverts,
            thrust::raw_pointer_cast(d_neighbor_offsets.data()),
            thrust::raw_pointer_cast(d_neighbor_indices.data()));
        timeP_s = t.seconds();
    }

    // Upload faces
    {
        std::vector<int3> h_faces(nfaces);
        for (int i = 0; i < nfaces; ++i)
            h_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_in_faces.data()),
                        h_faces.data(), nfaces * sizeof(int3),
                        cudaMemcpyHostToDevice, stream);
    }

    // Step 6: Frontier clustering loop
    thrust::device_vector<int> d_cp_vec(nverts);
    thrust::sequence(d_cp_vec.begin(), d_cp_vec.end());
    thrust::device_vector<int> d_active(nverts), d_next_active(nverts), d_next_count(1);

    int* h_next_count = nullptr;
    cudaMallocHost(&h_next_count, sizeof(int));

    int BLOCK_SIZE = 512, iter = 0;
    {
        GpuTimer total("Clustering phase", print_time, stream);

        auto end_it2 = thrust::copy_if(pol, first, last, d_depend.begin(), d_active.begin(),
                                        [] __device__(int dep){ return dep == 0; });
        int active_count = static_cast<int>(end_it2 - d_active.begin());

        while (active_count > 0) {
            thrust::fill(d_next_count.begin(), d_next_count.end(), 0);
            int numBlocks = ((active_count * 32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t shared_bytes = SHARED_FRONTIER_CAP * sizeof(int);

            strict_frontier_warp_blockBuffered_kernel<<<numBlocks, BLOCK_SIZE, shared_bytes, stream>>>(
                d_points,
                thrust::raw_pointer_cast(d_cp_vec.data()),
                thrust::raw_pointer_cast(d_depend.data()),
                thrust::raw_pointer_cast(d_active.data()),
                active_count,
                thrust::raw_pointer_cast(d_neighbor_indices.data()),
                thrust::raw_pointer_cast(d_neighbor_offsets.data()),
                nverts,
                thrust::raw_pointer_cast(d_next_active.data()),
                thrust::raw_pointer_cast(d_next_count.data()));

            if (enable_injection)
                total_pcie_seconds += gpuinj_do_copies(injectCopies, stream);

            int next = 0;
            cudaMemcpy(&next, thrust::raw_pointer_cast(d_next_count.data()),
                       sizeof(int), cudaMemcpyDeviceToHost);
            if (next > 0)
                thrust::copy_n(d_next_active.begin(), next, d_active.begin());
            active_count = next;
            iter++;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timeWhile_s = total.seconds();
    }
    std::cout << "        (iters = " << iter << ")\n";

    // Print cluster count
    {
        thrust::device_vector<int> d_is_cent(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0), thrust::make_counting_iterator(nverts),
            d_is_cent.begin(),
            [cp = thrust::raw_pointer_cast(d_cp_vec.data())] __device__(int i){
                return (cp[i] == i) ? 1 : 0; });
        int simplified = thrust::reduce(d_is_cent.begin(), d_is_cent.end(), 0, thrust::plus<int>());
        std::cout << "----------------------------------------------------------\n";
        std::cout << "Simplified vertices: " << simplified << " / " << nverts << "\n";
        std::cout << "----------------------------------------------------------\n";
    }

    // Steps 7-8: Compact IDs + Centroids + Face Remap
    int reduced_clusters = 0, valid_faces = 0;
    thrust::device_vector<double3> d_centroids;
    int3* h_in_faces = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in_faces, nfaces * sizeof(int3)));

    {
        GpuTimer t_single("Single region (centroids + remap)", print_time, stream);

        thrust::device_vector<int> d_is_centroid(nverts), d_prefix(nverts), d_pid2ccid(nverts);
        const int* cp_ptr     = thrust::raw_pointer_cast(d_cp_vec.data());
        const int* prefix_ptr = thrust::raw_pointer_cast(d_prefix.data());

        thrust::transform(thrust::cuda::par.on(stream), first, last, d_is_centroid.begin(),
            [cp_ptr] __device__(int i){ return (cp_ptr[i] == i) ? 1 : 0; });
        thrust::exclusive_scan(thrust::cuda::par.on(stream),
            d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());
        thrust::transform(thrust::cuda::par.on(stream), first, last, d_pid2ccid.begin(),
            [cp_ptr, prefix_ptr] __device__(int i){ return prefix_ptr[cp_ptr[i]]; });

        thrust::device_vector<int> d_keys(nverts);
        thrust::device_vector<SumCount> d_vals(nverts);
        thrust::copy_n(thrust::cuda::par.on(stream), d_pid2ccid.begin(), nverts, d_keys.begin());
        thrust::transform(thrust::cuda::par.on(stream), first, last, d_vals.begin(),
            [px, py, pz] __device__(int i){ return SumCount{ px[i], py[i], pz[i], 1 }; });
        thrust::sort_by_key(thrust::cuda::par.on(stream), d_keys.begin(), d_keys.end(), d_vals.begin());

        thrust::device_vector<int> d_ukeys(nverts);
        thrust::device_vector<SumCount> d_sums(nverts);
        auto new_end = thrust::reduce_by_key(thrust::cuda::par.on(stream),
            d_keys.begin(), d_keys.end(), d_vals.begin(),
            d_ukeys.begin(), d_sums.begin(),
            thrust::equal_to<int>(), SumCombine());
        reduced_clusters = static_cast<int>(new_end.first - d_ukeys.begin());

        d_centroids.resize(reduced_clusters);
        thrust::transform(thrust::cuda::par.on(stream),
            d_sums.begin(), d_sums.begin() + reduced_clusters, d_centroids.begin(),
            [] __device__(const SumCount& sc){
                double inv = 1.0 / (double)sc.count;
                return make_double3(sc.x*inv, sc.y*inv, sc.z*inv); });

        if (nfaces > 0) {
            GpuTimer t_mesh("Mesh update (face remap)", print_time, stream);
            d_out_faces.resize(nfaces);
            unsigned int* d_valid_counter = nullptr;
            cudaMalloc(&d_valid_counter, sizeof(unsigned int));
            cudaMemsetAsync(d_valid_counter, 0, sizeof(unsigned int), stream);

            remap_and_count_kernel<<<(nfaces+255)/256, 256, 0, stream>>>(
                thrust::raw_pointer_cast(d_in_faces.data()),
                thrust::raw_pointer_cast(d_out_faces.data()),
                thrust::raw_pointer_cast(d_pid2ccid.data()),
                nfaces, d_valid_counter);
            CUDA_CHECK(cudaGetLastError());

            {
                GpuTimer t_pcie2("PCIe D->H (face count)", print_time, stream);
                cudaMemcpyAsync(&valid_faces, d_valid_counter, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                cudaFree(d_valid_counter);
                auto end_rm = thrust::remove_if(thrust::cuda::par.on(stream),
                    d_out_faces.begin(), d_out_faces.end(),
                    [] __device__(const int3& f){ return (f.x < 0); });
                d_out_faces.resize(end_rm - d_out_faces.begin());
                pcie_d2h_s = t_pcie2.seconds();
            }
            timeUmesh_s = t_mesh.seconds();
            std::cout << "    (valid faces = " << valid_faces << ")\n";
        }
        timeSingle_s = t_single.seconds();
    }
    if (nfaces > 0 && h_in_faces) cudaFreeHost(h_in_faces);

    // Step 9: Copy results back to host
    {
        GpuTimer t_d2h("PCIe D->H (vertices + faces)", print_time, stream);
        thrust::host_vector<double3> h_out(d_centroids.begin(), d_centroids.end());
        vertices.clear(); vertices.reserve(h_out.size());
        for (auto& v : h_out) vertices.emplace_back(v.x, v.y, v.z);

        if (nfaces > 0) {
            thrust::host_vector<int3> h_faces(d_out_faces.begin(),
                                               d_out_faces.begin() + valid_faces);
            triangles.clear(); triangles.reserve(valid_faces);
            for (auto& f : h_faces) triangles.emplace_back(f.x, f.y, f.z);
            std::cout << "Copied back reduced mesh: "
                      << vertices.size() << " vertices, " << triangles.size() << " faces.\n";
        } else {
            triangles.clear();
            std::cout << "Copied back reduced cloud: " << vertices.size() << " vertices.\n";
        }
        pcie_d2h_s += t_d2h.seconds();
    }

    // Timing summary
    double grid_phase    = timeGrid_s + timeHash_s;
    double adj_phase     = timeCSR_s  + timeP_s;
    double cluster_phase = timeWhile_s + timeSingle_s + timeUmesh_s;
    double total_excl    = adj_phase + cluster_phase;
    double total_incl    = total_excl + pcie_h2d_s + pcie_d2h_s;

    std::cout << "\n----------------------------------------------------------\n";
    std::cout << "GPU Timing Summary (Model 2: GPU Streaming)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  PCIe H->D:                    " << pcie_h2d_s  << " s\n";
    std::cout << "  Indexing phase:               " << grid_phase  << " s\n";
    std::cout << "    Build hash grid:            " << timeGrid_s  << " s\n";
    std::cout << "    Build hash table:           " << timeHash_s  << " s\n";
    std::cout << "  Adjacency phase:              " << adj_phase   << " s\n";
    std::cout << "    Build neighbor list (CSR):  " << timeCSR_s   << " s\n";
    std::cout << "    Init depend[]:              " << timeP_s     << " s\n";
    std::cout << "  Clustering phase:             " << cluster_phase << " s\n";
    std::cout << "    While loop:                 " << timeWhile_s << " s\n";
    std::cout << "    Single region:              " << timeSingle_s << " s\n";
    std::cout << "    Face remap:                 " << timeUmesh_s << " s\n";
    std::cout << "  PCIe D->H:                    " << pcie_d2h_s  << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  TOTAL (excl PCIe): " << total_excl << " s\n";
    std::cout << "  TOTAL (incl PCIe): " << total_incl << " s\n";
    std::cout << "  Injected PCIe:     " << total_pcie_seconds << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Final: " << reduced_clusters << " clusters | "
              << valid_faces << " faces\n";

    cudaFreeHost(h_next_count);
    cudaFreeHost(h_points.x); cudaFreeHost(h_points.y); cudaFreeHost(h_points.z);
    cudaFree(d_points.x);     cudaFree(d_points.y);     cudaFree(d_points.z);
    if (enable_injection) gpuinj_free_buffers();
    cudaStreamDestroy(stream);
    std::cout << "GPU streaming clustering complete.\n";
}

// ============================================================
// Kernel: Init depend[] directly from hash grid (Model 1).
// For each vertex u, atomically increments depend[v] for all
// neighbors v > u within eps — no CSR precomputed.
// ============================================================
__global__ void init_depend_from_hash_kernel(
    const double3* __restrict__ points,
    int* __restrict__ depend,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    long long table_mask)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nverts) return;

    const double eps2 = eps * eps;
    double3 pu = points[u];
    int ix = (int)((pu.x - min_bound.x) / cell_size);
    int iy = (int)((pu.y - min_bound.y) / cell_size);
    int iz = (int)((pu.z - min_bound.z) / cell_size);

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        long long key = cell_key(ix+dx, iy+dy, iz+dz);
        int       idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
        if (idx < 0) continue;

        int start = cell_offsets[idx];
        int end   = cell_offsets[idx + 1];

        for (int k = start; k < end; k++) {
            int v = cell_points[k];
            if (v <= u) continue;

            double3 pv  = points[v];
            double  ddx = pu.x - pv.x;
            double  ddy = pu.y - pv.y;
            double  ddz = pu.z - pv.z;
            if (ddx*ddx + ddy*ddy + ddz*ddz <= eps2)
                atomicAdd(&depend[v], 1);
        }
    }
}

// ============================================================
// Model 1: On-the-Fly GPU Clustering
// Hash grid built once; neighbor search repeated each iteration.
// No precomputed CSR adjacency list.
// ============================================================
void merge_vertices_forward_gpu_otf(
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<Eigen::Vector3i>& triangles,
    double eps,
    bool print_time)
{
    int nverts = static_cast<int>(vertices.size());
    int nfaces = static_cast<int>(triangles.size());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto pol   = thrust::cuda::par.on(stream);
    auto first = thrust::make_counting_iterator<int>(0);
    auto last  = first + nverts;

    double timeGrid_s = 0, timeHash_s = 0, timeP_s = 0;
    double timeWhile_s = 0, timeSingle_s = 0, timeUmesh_s = 0;
    double pcie_h2d_s = 0, pcie_d2h_s = 0;

    // Step 1: Build GPU hash grid
    GridInfo grid;
    thrust::device_vector<long long> d_cell_keys;
    thrust::device_vector<int> d_cell_offsets, d_cell_points;
    {
        GpuTimer t("OTF: Build hash grid", print_time, stream);
        build_hash_grid_gpu_safe(vertices, eps, grid,
                                 d_cell_keys, d_cell_offsets, d_cell_points,
                                 print_time, stream);
        timeGrid_s = t.seconds();
    }

    // Step 2: Build GPU hash table
    int num_cells = static_cast<int>(d_cell_keys.size());
    int table_size = 1;
    while (table_size < num_cells * 4) table_size <<= 1;
    long long table_mask = table_size - 1;

    thrust::device_vector<long long> d_hash_keys(table_size, LLONG_MIN);
    thrust::device_vector<int>       d_hash_vals(table_size, -1);
    {
        GpuTimer t("OTF: Build hash table", print_time, stream);
        int threads = 256, blocks = (num_cells + threads - 1) / threads;
        build_hash_table_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_cell_keys.data()), num_cells,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()), table_mask);
        timeHash_s = t.seconds();
    }
    std::cout << "OTF hash table: " << table_size
              << " slots for " << num_cells << " cells.\n";

    // Step 3: Upload vertices H->D
    thrust::device_vector<double3> d_points(nverts);
    {
        GpuTimer t("OTF: PCIe H->D", print_time, stream);
        std::vector<double3> h_pts(nverts);
        for (int i = 0; i < nverts; ++i)
            h_pts[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_points.data()),
                        h_pts.data(), nverts * sizeof(double3),
                        cudaMemcpyHostToDevice, stream);
        pcie_h2d_s = t.seconds();
    }

    // Step 4: Init depend[] on-the-fly (no CSR)
    thrust::device_vector<int> d_depend(nverts, 0);
    thrust::device_vector<int> d_cp_vec(nverts);
    thrust::sequence(pol, d_cp_vec.begin(), d_cp_vec.end());

    const int N = num_cells;
    thrust::device_vector<int> d_occupied_indices(N);
    const int* offsets_ptr = thrust::raw_pointer_cast(d_cell_offsets.data());
    auto end_occ = thrust::copy_if(pol, first, first + N,
        d_occupied_indices.begin(),
        [offsets_ptr] __device__(int i){
            return (offsets_ptr[i+1] - offsets_ptr[i]) > 0; });
    int num_occupied = static_cast<int>(end_occ - d_occupied_indices.begin());
    d_occupied_indices.resize(num_occupied);

    {
        GpuTimer t("OTF: Init depend", print_time, stream);
        thrust::fill(pol, d_depend.begin(), d_depend.end(), 0);
        int threads_d = 256, blocks_d = (nverts + threads_d - 1) / threads_d;
        init_depend_from_hash_kernel<<<blocks_d, threads_d, 0, stream>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_depend.data()),
            nverts, eps, grid.min_bound, grid.cell_size,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()),
            thrust::raw_pointer_cast(d_cell_offsets.data()),
            thrust::raw_pointer_cast(d_cell_points.data()),
            table_mask);
        timeP_s = t.seconds();
    }

    // Step 5: Upload faces
    thrust::device_vector<int3> d_in_faces(nfaces), d_out_faces(nfaces);
    {
        std::vector<int3> h_faces(nfaces);
        for (int i = 0; i < nfaces; ++i)
            h_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_in_faces.data()),
                        h_faces.data(), nfaces * sizeof(int3),
                        cudaMemcpyHostToDevice, stream);
    }

    // Step 6: On-the-fly clustering loop
    thrust::device_vector<bool> d_changed(nverts, false);
    int iter = 0;
    {
        GpuTimer t_while("OTF: Clustering loop", print_time, stream);
        const int BLOCK    = 256;
        bool      any_changed = true;

        while (any_changed) {
            thrust::fill(pol, d_changed.begin(), d_changed.end(), false);

            int grid_dim = (nverts + BLOCK - 1) / BLOCK;
            streaming_iteration_hash_kernel<<<grid_dim, BLOCK, 0, stream>>>(
                thrust::raw_pointer_cast(d_points.data()),
                thrust::raw_pointer_cast(d_cp_vec.data()),
                thrust::raw_pointer_cast(d_depend.data()),
                thrust::raw_pointer_cast(d_changed.data()),
                nverts, eps, grid.min_bound, grid.cell_size,
                thrust::raw_pointer_cast(d_cell_keys.data()),
                thrust::raw_pointer_cast(d_cell_offsets.data()),
                thrust::raw_pointer_cast(d_cell_points.data()),
                num_cells,
                thrust::raw_pointer_cast(d_hash_keys.data()),
                thrust::raw_pointer_cast(d_hash_vals.data()),
                table_mask);
            CUDA_CHECK(cudaGetLastError());

            int changed = thrust::count_if(pol, d_changed.begin(), d_changed.end(),
                                            [] __device__(bool b){ return b; });
            any_changed = (changed > 0);
            iter++;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timeWhile_s = t_while.seconds();
    }
    std::cout << "OTF iters = " << iter << "\n";

    // Steps 7-8: Compact IDs + Centroids + Face Remap
    int reduced_clusters = 0, valid_faces = 0;
    thrust::device_vector<double3> d_centroids;

    {
        GpuTimer t_single("OTF: Single region", print_time, stream);

        const int* cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data());
        thrust::device_vector<int> d_is_centroid(nverts), d_prefix(nverts), d_pid2ccid(nverts);

        thrust::transform(pol, first, last, d_is_centroid.begin(),
            [cp_ptr] __device__(int i){ return (cp_ptr[i] == i) ? 1 : 0; });
        thrust::exclusive_scan(pol,
            d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());
        const int* prefix_ptr = thrust::raw_pointer_cast(d_prefix.data());
        thrust::transform(pol, first, last, d_pid2ccid.begin(),
            [cp_ptr, prefix_ptr] __device__(int i){ return prefix_ptr[cp_ptr[i]]; });

        thrust::device_vector<int>      d_keys(nverts);
        thrust::device_vector<SumCount> d_vals(nverts);
        thrust::copy_n(pol, d_pid2ccid.begin(), nverts, d_keys.begin());
        const double3* pts = thrust::raw_pointer_cast(d_points.data());
        thrust::transform(pol, first, last, d_vals.begin(),
            [pts] __device__(int i){
                double3 p = pts[i]; return SumCount{p.x, p.y, p.z, 1}; });
        thrust::sort_by_key(pol, d_keys.begin(), d_keys.end(), d_vals.begin());

        thrust::device_vector<int>      d_ukeys(nverts);
        thrust::device_vector<SumCount> d_sums(nverts);
        auto ne = thrust::reduce_by_key(pol,
            d_keys.begin(), d_keys.end(), d_vals.begin(),
            d_ukeys.begin(), d_sums.begin(),
            thrust::equal_to<int>(), SumCombine());
        reduced_clusters = static_cast<int>(ne.first - d_ukeys.begin());

        d_centroids.resize(reduced_clusters);
        thrust::transform(pol,
            d_sums.begin(), d_sums.begin() + reduced_clusters, d_centroids.begin(),
            [] __device__(const SumCount& sc){
                double inv = 1.0 / (double)sc.count;
                return make_double3(sc.x*inv, sc.y*inv, sc.z*inv); });

        if (nfaces > 0) {
            GpuTimer t_mesh("OTF: Face remap", print_time, stream);
            unsigned int* d_valid_ctr = nullptr;
            cudaMalloc(&d_valid_ctr, sizeof(unsigned int));
            cudaMemsetAsync(d_valid_ctr, 0, sizeof(unsigned int), stream);

            remap_and_count_kernel<<<(nfaces+255)/256, 256, 0, stream>>>(
                thrust::raw_pointer_cast(d_in_faces.data()),
                thrust::raw_pointer_cast(d_out_faces.data()),
                thrust::raw_pointer_cast(d_pid2ccid.data()),
                nfaces, d_valid_ctr);
            CUDA_CHECK(cudaGetLastError());

            cudaMemcpyAsync(&valid_faces, d_valid_ctr,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaFree(d_valid_ctr);

            auto rm_end = thrust::remove_if(pol, d_out_faces.begin(), d_out_faces.end(),
                [] __device__(const int3& f){ return (f.x < 0); });
            d_out_faces.resize(rm_end - d_out_faces.begin());
            timeUmesh_s = t_mesh.seconds();
            std::cout << "OTF valid faces = " << valid_faces << "\n";
        }
        timeSingle_s = t_single.seconds();
    }

    // Step 9: Copy results back to host
    {
        GpuTimer t_d2h("OTF: PCIe D->H", print_time, stream);
        thrust::host_vector<double3> h_verts(d_centroids.begin(), d_centroids.end());
        vertices.clear(); vertices.reserve(h_verts.size());
        for (auto& v : h_verts) vertices.emplace_back(v.x, v.y, v.z);

        if (nfaces > 0) {
            thrust::host_vector<int3> h_faces(d_out_faces.begin(),
                                               d_out_faces.begin() + valid_faces);
            triangles.clear(); triangles.reserve(valid_faces);
            for (auto& f : h_faces) triangles.emplace_back(f.x, f.y, f.z);
        } else {
            triangles.clear();
        }
        pcie_d2h_s = t_d2h.seconds();
    }

    // Timing summary
    double total_excl = timeP_s + timeWhile_s + timeSingle_s + timeUmesh_s;
    double total_incl = total_excl + pcie_h2d_s + pcie_d2h_s;

    std::cout << "\n----------------------------------------------------------\n";
    std::cout << "OTF GPU Timing Summary (Model 1)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  PCIe H->D:          " << pcie_h2d_s   << " s\n";
    std::cout << "  Build hash grid:    " << timeGrid_s   << " s\n";
    std::cout << "  Build hash table:   " << timeHash_s   << " s\n";
    std::cout << "  Init depend (OTF):  " << timeP_s      << " s\n";
    std::cout << "  Clustering loop:    " << timeWhile_s  << " s\n";
    std::cout << "  Single region:      " << timeSingle_s << " s\n";
    std::cout << "  Face remap:         " << timeUmesh_s  << " s\n";
    std::cout << "  PCIe D->H:          " << pcie_d2h_s  << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  TOTAL (excl PCIe):  " << total_excl  << " s\n";
    std::cout << "  TOTAL (incl PCIe):  " << total_incl  << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Final: " << vertices.size() << " vertices, "
              << triangles.size() << " faces\n";
    std::cout << "OTF GPU clustering complete.\n";

    cudaStreamDestroy(stream);
}

} // namespace geometry
} // namespace open3d
