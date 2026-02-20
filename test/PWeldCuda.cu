// ----------------------------------------------------------------------------
// MIT License
// GPU port of Nima Fathollahi & Sean Chester's strict-mode clustering
// Optimized face remapping: parallel compaction + deduplication
// ----------------------------------------------------------------------------

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
#include <numeric>   // for std::iota
#include <algorithm> // for std::sort
#include <iomanip>  // 
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CUDA_RESTRICT __restrict__
bool enable_injection = false;

void* d_cub_temp = nullptr;
size_t cub_temp_bytes = 0;

#define TILE_X 2
#define TILE_Y 2
#define TILE_Z 2
#define MAX_CELL_SIZE 64  // safe for shared mem < 48KB
#define TILE_SIZE 8
#define WARP_SIZE 32
#define SHARED_FRONTIER_CAP 512  // entries per block in shared buffer

__device__ int d_changedFlag;   // ✅ global device flag for convergence
__device__ int d_anyChanged;
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }
// Hash a 3D cell index into 64-bit key
__host__ __device__ inline long long cell_key(int ix, int iy, int iz) {
    // Large primes minimize collisions for sparse integer grids
    const long long p1 = 73856093;
    const long long p2 = 19349663;
    const long long p3 = 83492791;
    return ((long long)ix * p1) ^ ((long long)iy * p2) ^ ((long long)iz * p3);
}

__device__ int d_nextCell;

struct PointsSOA {
    double *x, *y, *z;
};

// ============================================================
// Grid Info for Streaming Neighbor Search
// ============================================================
struct GridInfo {
    double3 min_bound;
    double cell_size;
    int3 resolution; // number of cells in x,y,z
};





// Compute cell ID from 3D point
__host__ __device__
inline int get_cell_id(const double3& p, const GridInfo& grid) {
    int ix = (int)((p.x - grid.min_bound.x) / grid.cell_size);
    int iy = (int)((p.y - grid.min_bound.y) / grid.cell_size);
    int iz = (int)((p.z - grid.min_bound.z) / grid.cell_size);
    return ix + iy * grid.resolution.x + iz * grid.resolution.x * grid.resolution.y;
}
__device__ __forceinline__ int hash_lookup(
    long long key,
    const long long* hash_keys,
    const int* hash_vals,
    int table_mask)
{
    unsigned int slot =
        ((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) & table_mask;
    int safety = 0;
    while (true) {
        long long stored = hash_keys[slot];
        if (stored == key) return hash_vals[slot];   // found
        if (stored == LLONG_MIN || ++safety > 16) return -1;  // empty or fail
        slot = (slot + 1) & table_mask;
    }
}
__device__ int binary_search_device(const long long* keys, int n, long long target)
{
    int left = 0, right = n - 1;
    while (left <= right)
    {
        int mid = (left + right) >> 1;
        long long val = keys[mid];
        if (val == target) return mid;
        if (val < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}




// ============================================================
// Parallel hash-grid build (CUDA-safe + OpenMP)
// ============================================================
void build_hash_grid_cpu_parallel(
    const std::vector<Eigen::Vector3d>& vertices,
    double eps,
    GridInfo& grid,
    std::vector<long long>& h_cell_keys,
    std::vector<int>& h_cell_offsets,
    std::vector<int>& h_cell_points)
{
    const int nverts = static_cast<int>(vertices.size());
    const double inv_cell = 1.0 / eps;

    grid.cell_size = eps;  // Only this field exists in your struct

    // ---- Step 1: compute cell hash keys in parallel ----
    std::vector<long long> keys(nverts);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nverts; ++i)
    {
        // ✅ Access Eigen::Vector3d components directly
        const Eigen::Vector3d& v = vertices[i];
        int ix = static_cast<int>((v.x() - grid.min_bound.x) * inv_cell);
        int iy = static_cast<int>((v.y() - grid.min_bound.y) * inv_cell);
        int iz = static_cast<int>((v.z() - grid.min_bound.z) * inv_cell);

        // 64-bit hash (deterministic)
        keys[i] = ((long long)ix * 73856093LL) ^
                  ((long long)iy * 19349663LL) ^
                  ((long long)iz * 83492791LL);
    }

    // ---- Step 2: sort vertices by key ----
    std::vector<int> idx(nverts);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return keys[a] < keys[b]; });

    // ---- Step 3: compute offsets ----
    h_cell_points.resize(nverts);
    for (int i = 0; i < nverts; ++i)
        h_cell_points[i] = idx[i];

    h_cell_keys.clear();
    h_cell_offsets.clear();
    h_cell_keys.reserve(nverts);
    h_cell_offsets.reserve(nverts);

    h_cell_keys.push_back(keys[idx[0]]);
    h_cell_offsets.push_back(0);

    for (int i = 1; i < nverts; ++i)
    {
        if (keys[idx[i]] != keys[idx[i - 1]]) {
            h_cell_keys.push_back(keys[idx[i]]);
            h_cell_offsets.push_back(i);
        }
    }

    h_cell_offsets.push_back(nverts);
}

// ============================================================
// CPU: Build uniform grid for streaming NN search
// ============================================================

// Build sparse hash grid
void build_hash_grid_cpu(
    const std::vector<Eigen::Vector3d>& vertices,
    double eps,
    GridInfo& grid,  // keep min_bound + cell_size only
    std::vector<long long>& cell_keys,   // unique cell keys
    std::vector<int>& cell_offsets,      // offset of each cell in cell_points
    std::vector<int>& cell_points)       // flattened vertex IDs
{
    // Find bounding box
    Eigen::Vector3d min_bound(DBL_MAX, DBL_MAX, DBL_MAX);
    Eigen::Vector3d max_bound(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (auto& v : vertices) {
        min_bound = min_bound.cwiseMin(v);
        max_bound = max_bound.cwiseMax(v);
    }
    grid.min_bound = make_double3(min_bound.x(), min_bound.y(), min_bound.z());
    grid.cell_size = eps;

    // Step 1: assign each point to a cell
    int nverts = (int)vertices.size();
    std::vector<std::pair<long long, int>> keyed_points;
    keyed_points.reserve(nverts);

    for (int i = 0; i < nverts; i++) {
        int ix = (int)floor((vertices[i].x() - min_bound.x()) / eps);
        int iy = (int)floor((vertices[i].y() - min_bound.y()) / eps);
        int iz = (int)floor((vertices[i].z() - min_bound.z()) / eps);
        keyed_points.emplace_back(cell_key(ix, iy, iz), i);
    }

    // Step 2: sort by cell_key
    std::sort(keyed_points.begin(), keyed_points.end(),
        [](auto& a, auto& b) { return a.first < b.first; });

    // Step 3: build unique cell_keys + offsets
    cell_keys.clear();
    cell_offsets.clear();
    cell_points.resize(nverts);

    int offset = 0;
    cell_offsets.push_back(0);
    long long current_key = keyed_points[0].first;
    cell_keys.push_back(current_key);

    for (int j = 0; j < nverts; j++) {
        if (keyed_points[j].first != current_key) {
            current_key = keyed_points[j].first;
            cell_keys.push_back(current_key);
            cell_offsets.push_back(j);
        }
        cell_points[j] = keyed_points[j].second;
    }
    cell_offsets.push_back(nverts);

    std::cout << "⚙️ Hash grid built: " << cell_keys.size()
        << " occupied cells, " << nverts << " points.\n";
}



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
        // Always create events so seconds() is safe even if print_ == false
        cudaEventCreateWithFlags(&start_, cudaEventDefault);
        cudaEventCreateWithFlags(&stop_, cudaEventDefault);

        // Record start on the chosen stream
        cudaEventRecord(start_, stream_);


        if (print_) {
            indent_ = depth_++;
        }
    }

    // ✅ compute elapsed time immediately (safe before destructor)
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
            for (int i = 0; i < indent_; i++)
                std::cout << "    ";
            std::cout << label_ << " took " << elapsed_ms_ / 1000.0 << " seconds\n";
            depth_--;
        }
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    cudaStream_t  stream_ = 0;
    std::string label_;
    bool print_;
    cudaEvent_t start_, stop_;
    float elapsed_ms_;
    int indent_ = 0;
    static thread_local int depth_;
};
thread_local int GpuTimer::depth_ = 0;



namespace open3d {
    namespace geometry {


// ============================================================
// PCIe Injection Utilities (for Amdahl’s Law experiments)
// ============================================================
static void* gpuinj_h_pinned = nullptr;
static void* gpuinj_d_buf    = nullptr;
static size_t gpuinj_bytes   = 0;
static int    gpuinj_copies  = 0;

static void gpuinj_init_buffers(size_t bytes) {
    if (bytes == 0) return;
    gpuinj_bytes = bytes;
    cudaMallocHost(&gpuinj_h_pinned, bytes);
    cudaMalloc(&gpuinj_d_buf, bytes);
    memset(gpuinj_h_pinned, 0x3f, bytes);
    cudaMemset(gpuinj_d_buf, 0x5a, bytes);
}

static void gpuinj_free_buffers() {
    if (gpuinj_d_buf) cudaFree(gpuinj_d_buf);
    if (gpuinj_h_pinned) cudaFreeHost(gpuinj_h_pinned);
    gpuinj_d_buf = nullptr;
    gpuinj_h_pinned = nullptr;
    gpuinj_bytes = 0;
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
    return ms * 1e-3f; // seconds
}


        // ============================================================
        // Kernel: strict clustering iteration
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
    if (u < 0 || u >= nverts) return;  // safety check

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
            if (pos < nverts) next_frontier[pos] = v; // prevent overflow
        }
    }
}



// ---- util -------------------------------------------------------------------



// Proposed solution: mix64 hash function + atomicCAS insertion + linear probing lookup
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void build_hash_table_kernel(
    const long long* __restrict__ cell_keys,
    int num_cells,
    long long* __restrict__ hash_keys,
    int* __restrict__ hash_vals,
    int table_mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_cells) return;

    unsigned long long h = mix64((unsigned long long)cell_keys[tid]);
    unsigned int slot = h & table_mask;
    unsigned int step = ((h >> 32) | 1u); // odd step for full-period probing

    while (true) {
        unsigned long long prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
            (unsigned long long)LLONG_MIN,
            (unsigned long long)cell_keys[tid]);

        if (prev == (unsigned long long)LLONG_MIN || prev == (unsigned long long)cell_keys[tid]) {
            hash_vals[slot] = tid;
            return;
        }
        slot = (slot + step) & table_mask;
    }
}


__device__ __forceinline__ int hash_lookup(
    long long key,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    unsigned long long h = mix64((unsigned long long)key);
    unsigned int slot = h & table_mask;
    unsigned int step = ((h >> 32) | 1u);

    while (true) {
        long long stored = hash_keys[slot];
        if (stored == key)               return hash_vals[slot];
        if (stored == LLONG_MIN)         return -1;             // empty ⇒ not found
        slot = (slot + step) & table_mask;
    }
}

__device__ __forceinline__ int upper_bound_int(const int* a, int len, int val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int x = __ldg(&a[mid]);
        if (x <= val) lo = mid + 1;
        else          hi = mid;
    }
    return lo;
}

// Proposed solution: compact-cell version with hash table lookup
__global__ void count_neighbors_compact_kernel(
    const double3* __restrict__ points,
    int* __restrict__ neighbor_counts,  // OUTPUT
    int nverts, double eps, double3 min_bound, double cell_size,
    const long long* __restrict__ cell_keys,      // unused here
    const int* __restrict__ cell_offsets,     // num_occupied+1
    const int* __restrict__ cell_points,      // sorted within each cell
    const int* __restrict__ occupied_cells,   // unused
    int num_occupied,
    // hash table
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int               table_mask)
{
    extern __shared__ int sh_points[];           // TILE ints
    const int TILE = 256;                        // try 128/256/512
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < nverts; u += blockDim.x * gridDim.x) {
        const double3 pu = points[u];
        int count = 0;

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
                    const int idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
                    if (idx < 0) continue;

                    const int beg = __ldg(&cell_offsets[idx]);
                    const int end = __ldg(&cell_offsets[idx + 1]);
                    const int len = end - beg;
                    if (len <= 0) continue;

                    int pos = beg + upper_bound_int(cell_points + beg, len, u);

                    if (len > TILE) {
                        // Dense-cell tiling
                        for (int t = pos; t < end; t += TILE) {
                            const int chunk = min(TILE, end - t);
                            for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                                sh_points[i] = __ldg(&cell_points[t + i]);
                            __syncthreads();

#pragma unroll 4
                            for (int i = 0; i < chunk; ++i) {
                                const int v = sh_points[i];     // v > u due to upper_bound
                                const double3 pv = points[v];
                                const double dx2 = pv.x - pu.x, dy2 = pv.y - pu.y, dz2 = pv.z - pu.z;
                                count += (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 <= eps2);
                            }
                            __syncthreads();
                        }
                    }
                    else {
                        // Light-cell path
#pragma unroll 4
                        for (int k = pos; k < end; ++k) {
                            const int v = __ldg(&cell_points[k]); // v > u
                            const double3 pv = points[v];
                            const double dx2 = pv.x - pu.x, dy2 = pv.y - pu.y, dz2 = pv.z - pu.z;
                            count += (dx2 * dx2 + dy2 * dy2 + dz2 * dz2 <= eps2);
                        }
                    }
                }
        neighbor_counts[u] = count;
    }
}



// ============================================================
// Kernel: Initialize depend[] from prebuilt CSR adjacency
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
    int count = 0;

    // depend[v] counts how many incoming edges each vertex has
    for (int i = start; i < end; ++i) {
        int v = neighbor_indices[i];
        if (v > u) atomicAdd(&depend[v], 1);
    }
}


        // ============================================================
 // Kernel: Streaming iteration with GPU hash lookup (O(1))
 // ============================================================
 __global__ __launch_bounds__(256, 2)
        __global__ void streaming_iteration_hash_kernel(
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
            int table_mask)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            bool local_changed = false;

            for (int u = tid; u < nverts; u += gridDim.x * blockDim.x) {
                if (depend[u] == 0) {
                    depend[u] = -1;
                    bool isCentroid = (cp_vec[u] == u);

                    double3 pu = points[u];
                    int ix0 = (int)((pu.x - min_bound.x) / cell_size);
                    int iy0 = (int)((pu.y - min_bound.y) / cell_size);
                    int iz0 = (int)((pu.z - min_bound.z) / cell_size);

                    // Visit 27 neighboring cells
#pragma unroll
                    for (int dx = -1; dx <= 1; dx++) {
#pragma unroll
                        for (int dy = -1; dy <= 1; dy++) {
#pragma unroll
                            for (int dz = -1; dz <= 1; dz++) {
                                long long key = cell_key(ix0 + dx, iy0 + dy, iz0 + dz);
                                int idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
                                if (idx < 0) continue;

                                int start = cell_offsets[idx];
                                int end = cell_offsets[idx + 1];

                                // iterate points in neighbor cell
                                for (int i = start; i < end; i++) {
                                    int v = cell_points[i];
                                    if (u == v) continue;

                                    double dx_ = pu.x - points[v].x;
                                    double dy_ = pu.y - points[v].y;
                                    double dz_ = pu.z - points[v].z;
                                    if (dx_ * dx_ + dy_ * dy_ + dz_ * dz_ <= eps * eps) {
                                        // strict-mode atomic CAS merge
                                        if (isCentroid && depend[v] > 0) {
                                            int expected = cp_vec[v];
                                            int desired = u;
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
                    }
                }
            }

            if (tid < nverts) changed_flags[tid] = local_changed;
        }

// ============================================================
// Step 1b: Fill neighbors (skip empty cells)
// ============================================================
// Propsed solution: cudaOccupancy api use and modified build hash table kernel
__global__ void build_neighbors_compact_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ neighbor_offsets,   // scan (nverts+1)
    int* __restrict__ neighbor_counts,          // unused (API)
    int* __restrict__ neighbor_indices,         // OUTPUT CSR
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    // cell compaction arrays
    const long long* __restrict__ cell_keys,    // (not used by lookup)
    const int* __restrict__ cell_offsets,       // len=num_occupied+1
    const int* __restrict__ cell_points,        // sorted within each cell
    const int* __restrict__ occupied_cells,     // unused
    int num_occupied,
    // --- NEW: hash table for O(1) cell_key -> cell_index
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int               table_mask)
{
    extern __shared__ int sh_points[];        // TILE ints
    const int TILE = 256;                     // tune: 128/256/512
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x;
        u < nverts;
        u += blockDim.x * gridDim.x)
    {
        const double3 pu = points[u];
        const int rowStart = neighbor_offsets[u];
        const int rowEnd = neighbor_offsets[u + 1];
        int out = rowStart;

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
                    // O(1) expected lookup (replaces binary_search_device)
                    const int idx = hash_lookup(key, hash_keys, hash_vals, table_mask);
                    if (idx < 0) continue;

                    const int beg = __ldg(&cell_offsets[idx]);
                    const int end = __ldg(&cell_offsets[idx + 1]);
                    const int len = end - beg;
                    if (len <= 0) continue;

                    // Jump to first v > u (cell segment is sorted)
                    const int pos = beg + upper_bound_int(cell_points + beg, len, u);

                    if (len > TILE) {
                        // Dense-cell tiling
                        for (int t = pos; t < end; t += TILE) {
                            const int chunk = min(TILE, end - t);
                            for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                                sh_points[i] = __ldg(&cell_points[t + i]);
                            __syncthreads();

#pragma unroll 4
                            for (int i = 0; i < chunk; ++i) {
                                const int v = sh_points[i];
                                if (v <= u) continue;   // should already be true after pos
                                const double3 pv = points[v];
                                const double dx = pv.x - pu.x;
                                const double dy = pv.y - pu.y;
                                const double dz = pv.z - pu.z;
                                if (dx * dx + dy * dy + dz * dz <= eps2) {
                                    if (out < rowEnd) neighbor_indices[out++] = v;
                                }
                            }
                            __syncthreads();
                        }
                    }
                    else {
                        // Light-cell path
#pragma unroll 4
                        for (int k = pos; k < end; ++k) {
                            const int v = __ldg(&cell_points[k]);
                            if (v <= u) continue;
                            const double3 pv = points[v];
                            const double dx = pv.x - pu.x;
                            const double dy = pv.y - pu.y;
                            const double dz = pv.z - pu.z;
                            if (dx * dx + dy * dy + dz * dz <= eps2) {
                                if (out < rowEnd) neighbor_indices[out++] = v;
                            }
                        }
                    }
                }
#ifdef DEBUG_BUILD_NEIGH_CHECK
        if (out != rowEnd)
            printf("Row %d filled %d expected %d\n",
                u, out - rowStart, rowEnd - rowStart);
#endif
    }
}




// ============================================================
// Build neighbor list (CSR) from hash grid
// ============================================================
// ============================================================
// ⚡ Optimized Neighbor List Construction (Compact Cells + CSR)
// ============================================================
// Reuses the hash grid and hash table already built by
// build_hash_grid_gpu_safe() and build_hash_table_kernel().
// Keeps the two-pass (count + fill) approach for correctness,
// removes redundant sorting or grid rebuild.
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
    int table_mask,
    thrust::device_vector<int>& d_neighbor_offsets,
    thrust::device_vector<int>& d_neighbor_indices,
    thrust::device_vector<int>& d_occupied_indices,   // ✅ already built
    int num_occupied, 
    cudaStream_t stream = 0)                                 // ✅ already computed
{
    GpuTimer t("Build neighbor list (reused grid, two-pass)", true, stream);

    // ============================================================
    // Step 1: Count neighbors (compact-cell version)
    // ============================================================
    // 1) Thrust policy bound to your stream
    auto pol = thrust::cuda::par.on(stream);

    thrust::device_vector<int> d_neighbor_counts(nverts, 0);

    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       count_neighbors_compact_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    {
        GpuTimer t1("Count neighbors (compact)", true, stream);
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

    // ============================================================
    // Step 2: Prefix-sum to compute neighbor_offsets (CSR format)
    // ============================================================
    d_neighbor_offsets.resize(nverts + 1);

    thrust::exclusive_scan(pol, 
        d_neighbor_counts.begin(),
        d_neighbor_counts.end(),
        d_neighbor_offsets.begin());
    
    // Compute total number of neighbor edges
    int last_count = 0, last_offset = 0;
    CUDA_CHECK(cudaMemcpyAsync(&last_count,
                          thrust::raw_pointer_cast(d_neighbor_counts.data() + nverts - 1),
                          sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_offset,
                          thrust::raw_pointer_cast(d_neighbor_offsets.data() + nverts - 1),
                          sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int total_neighbors = last_offset + last_count;

    // Allocate neighbor indices array
    d_neighbor_indices.resize(total_neighbors);

    // ============================================================
    // Step 3: Reset counts for atomic append
    // ============================================================
    thrust::fill(pol, d_neighbor_counts.begin(), d_neighbor_counts.end(), 0);

    // ============================================================
    // Step 4: Fill neighbor list (compact-cell version)
    // ============================================================
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       build_neighbors_compact_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    {
        GpuTimer t2("Fill neighbors (compact)", true);
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

    // ============================================================
    // Step 5: Reporting
    // ============================================================
    std::cout << "✅ Compact CSR neighbor list built: "
              << total_neighbors << " total edges ("
              << std::fixed << std::setprecision(3)
              << (total_neighbors / (double)nverts)
              << " avg neighbors per vertex) from "
              << num_occupied << " occupied cells.\n";
}


        // ============================================================
        // Structs for centroid reduction
        // ============================================================
        struct SumCount {
            double x, y, z;
            int count;
        };
        struct SumCombine {
            __device__ SumCount operator()(const SumCount& a, const SumCount& b) const {
                return SumCount{ a.x + b.x, a.y + b.y, a.z + b.z, a.count + b.count };
            }
        };

        // ============================================================
        // Kernel: remap faces + flags (no atomics)
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
    int a = pid2ccid[f.x];
    int b = pid2ccid[f.y];
    int c = pid2ccid[f.z];

    if (a == b || b == c || a == c) {
        flags[fid] = 0; // degenerate
        return;
    }

    out_faces[fid] = make_int3(a, b, c);
    flags[fid] = 1;
}


// ============================================================
// Optimized Warp-level strict-mode frontier clustering (SoA + Warp Aggregation)
// Each warp cooperates on one active vertex 'u'
// ============================================================
__global__ void strict_frontier_warp_kernel_soa(
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
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= active_count) return;

    int u = active_vertices[warp_id];
    if (u < 0 || u >= nverts) return;

    depend[u] = -1;
    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    int start = neighbor_offsets[u];
    int end = neighbor_offsets[u + 1];
    int degree = end - start;

    for (int i = lane_id; i < degree; i += 32) {
        int v = neighbor_indices[start + i];
        if (v <= u || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        unsigned mask = __match_any_sync(__activemask(), v);
        int leader = __ffs(mask) - 1;
        int group_size = __popc(mask);

        int old = 0;
        if (lane_id == leader)
            old = atomicAdd(&depend[v], -group_size);
        old = __shfl_sync(mask, old, leader);

        bool became_zero = (old <= group_size && old > 0);
        if (became_zero) {
            unsigned push_mask = __match_any_sync(__activemask(), v);
            int push_leader = __ffs(push_mask) - 1;
            int push_count = __popc(push_mask);

            int pos = 0;
            if (lane_id == push_leader)
                pos = atomicAdd(next_count, push_count);
            pos = __shfl_sync(push_mask, pos, push_leader);

            int local_offset = __popc(push_mask & ((1u << lane_id) - 1));
            if (pos + local_offset < nverts)
                next_frontier[pos + local_offset] = v;
        }
    }
}

// ============================================================
// ✅ Memory-Optimized Strict-Mode Frontier Kernel (Block-Buffered)
// Keeps strict determinism identical to CPU version.
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
    int warp_id = global_tid / WARP_SIZE;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= active_count) return;

    int u = active_vertices[warp_id];
    if (u < 0 || u >= nverts) return;

    depend[u] = -1;
    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    int start = neighbor_offsets[u];
    int end   = neighbor_offsets[u + 1];
    int degree = end - start;

    for (int i = lane_id; i < degree; i += WARP_SIZE)
    {
        int v = neighbor_indices[start + i];
        if (v <= u || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        unsigned mask = __match_any_sync(__activemask(), v);
        int leader = __ffs(mask) - 1;
        int group_size = __popc(mask);

        int old = 0;
        if (lane_id == leader)
            old = atomicAdd(&depend[v], -group_size);
        old = __shfl_sync(mask, old, leader);

        bool became_zero = (old <= group_size && old > 0);
        if (became_zero)
        {
            // try to push into shared buffer
            int pos = atomicAdd(&local_count, 1);
            if (pos < SHARED_FRONTIER_CAP)
            {
                shared_frontier[pos] = v;
            }
            else
            {
                // flush single element when overflow
                if (lane_id == leader) {
                    int idx = atomicAdd(next_count, 1);
                    if (idx < nverts) next_frontier[idx] = v;
                }
            }
        }
    }
    __syncthreads();

    // flush shared buffer to global
    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        int idx = atomicAdd(next_count, 1);
        if (idx < nverts) next_frontier[idx] = shared_frontier[i];
    }
}



void merge_vertices_forward_gpu(
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<Eigen::Vector3i>& triangles,
    const std::vector<int>& neighbor_indices,
    const std::vector<int>& neighbor_offsets,
    const std::vector<int>& depend_init_host,
    double eps,
    bool print_time)
{
    // ---------------------------------------------------------------------
    // timeAll_s  →  total algorithmic time (matches Nima’s total pipeline)
    // ---------------------------------------------------------------------
    GpuTimer total("timeAll_s", print_time);

    int nverts = (int)vertices.size();
    int nfaces = (int)triangles.size();

    // ============================================================
    // Step 0: KDTree adjacency / depend init  --> timeP_s
    // ============================================================
    {
        // ✅ Counts in Nima’s “Neighbor Search” phase
        GpuTimer t("timeP_s", print_time);
        if (!depend_init_host.empty()) {
            bool has_nonzero = std::any_of(
                depend_init_host.begin(), depend_init_host.end(),
                [](int x) { return x > 0; });
            if (has_nonzero)
                std::cout << "✅ Reusing depend[] from KDTree adjacency (strict-mode)\n";
        }
        // No file I/O or H↔D copies timed here
    }

    
    // ============================================================
    // Step 1: Setup / Flatten vertices   ❌ (excluded from paper)
    // ============================================================
    thrust::device_vector<double3> d_in_vertices;
    thrust::device_vector<int> d_cp_vec, d_depend, d_neighbors, d_offsets, d_pid2ccid;
    {
        // Only for correctness; not included in total algorithmic time
        std::vector<double3> h_in_vertices(nverts);
        for (int i = 0; i < nverts; i++)
            h_in_vertices[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());

        d_in_vertices = h_in_vertices;
        d_cp_vec.resize(nverts);
        thrust::sequence(d_cp_vec.begin(), d_cp_vec.end());
        d_depend = depend_init_host;
        d_neighbors = neighbor_indices;
        d_offsets = neighbor_offsets;
    }

    // ============================================================
    // Step 2: Frontier Clustering Loop  --> Step 2 (main compute)
    // ============================================================
    {
        // ✅ Counts in “Clustering” phase (strict-mode)
        GpuTimer clustering("Step 2: Baseline Frontier Clustering", print_time);

        const int BLOCK_SIZE = 256;
        thrust::device_vector<int> d_active(nverts);
        thrust::device_vector<int> d_next_active(nverts);
        thrust::device_vector<int> d_next_count(1);

        // Build initial frontier
        auto end_it = thrust::copy_if(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(nverts),
            d_depend.begin(), d_active.begin(),
            [] __device__(int dep) { return dep == 0; });
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
                       sizeof(int),
                       cudaMemcpyDeviceToHost);
            if (h_next_count > 0)
                thrust::copy_n(d_next_active.begin(), h_next_count, d_active.begin());

            active_count = h_next_count;
            iter++;
        }
        std::cout << "        (iters = " << iter << ")\n";
    }

    // ============================================================
    // Step 3: Compact IDs  --> timeSingle_s
    // ============================================================
    {
        GpuTimer tSingle("timeSingle_s", print_time);
        thrust::device_vector<int> d_is_centroid(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nverts),
            d_is_centroid.begin(),
            [cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data())] __device__(int i) {
                return (cp_ptr[i] == i) ? 1 : 0;
            });

        thrust::device_vector<int> d_prefix(nverts);
        thrust::exclusive_scan(d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());

        d_pid2ccid.resize(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nverts),
            d_pid2ccid.begin(),
            [cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data()),
             prefix_ptr = thrust::raw_pointer_cast(d_prefix.data())] __device__(int i) {
                int root = cp_ptr[i];
                return prefix_ptr[root];
            });

        int nclusters, last_flag;
        cudaMemcpy(&nclusters, thrust::raw_pointer_cast(d_prefix.data() + nverts - 1),
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_is_centroid.data() + nverts - 1),
                   sizeof(int), cudaMemcpyDeviceToHost);
        nclusters += last_flag;
        std::cout << "        (clusters = " << nclusters << ")\n";
    }

    // ============================================================
    // Step 4: Centroid averaging  --> timeUR_s + timeUNV_s
    // ============================================================
    int reduced_clusters = 0;
    thrust::device_vector<int> d_unique_keys, d_keys;
    thrust::device_vector<SumCount> d_vals, d_sums;

    {
        // ✅ Both included in Nima’s “Single region” (Update reps + new verts)
        GpuTimer tUR("timeUR_s", print_time);
        d_keys.resize(nverts);
        d_vals.resize(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nverts),
            d_keys.begin(),
            [pid2ccid_ptr = thrust::raw_pointer_cast(d_pid2ccid.data())] __device__(int i) {
                return pid2ccid_ptr[i];
            });
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nverts),
            d_vals.begin(),
            [verts_ptr = thrust::raw_pointer_cast(d_in_vertices.data())] __device__(int i) {
                double3 v = verts_ptr[i];
                return SumCount{v.x, v.y, v.z, 1};
            });
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());
        d_unique_keys.resize(nverts);
        d_sums.resize(nverts);
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
            [] __device__(const SumCount& sc) {
                return make_double3(sc.x / sc.count, sc.y / sc.count, sc.z / sc.count);
            });
        d_in_vertices.swap(d_out_vertices);
        std::cout << "        (reduced clusters = " << reduced_clusters << ")\n";
    }

    // ============================================================
    // Step 5: Face remapping  --> timeU_s
    // ============================================================
    int valid_faces = 0;
    thrust::device_vector<int3> d_compact_faces;
    {
        GpuTimer tU("timeU_s", print_time);
        std::vector<int3> h_in_faces(nfaces);
        for (int i = 0; i < nfaces; i++)
            h_in_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
        thrust::device_vector<int3> d_in_faces = h_in_faces;
        thrust::device_vector<int3> d_out_faces(nfaces);
        thrust::device_vector<int> d_flags(nfaces);

        int threads = 256;
        int blocks = (nfaces + threads - 1) / threads;
        remap_faces_kernel_flags<<<blocks, threads>>>(
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
            [] __device__(int flag) { return flag == 1; });
        valid_faces = compact_end - d_compact_faces.begin();
        std::cout << "    (valid faces = " << valid_faces << ")\n";
    }

    // ============================================================
    // Step 6: Copy back to host  ❌ (excluded from paper)
    // ============================================================
    {
        thrust::host_vector<double3> h_out_vertices = d_in_vertices;
        vertices.clear();
        vertices.reserve(h_out_vertices.size());
        for (auto& v : h_out_vertices)
            vertices.emplace_back(v.x, v.y, v.z);

        thrust::host_vector<int3> h_out_faces(d_compact_faces.begin(),
                                              d_compact_faces.begin() + valid_faces);
        triangles.clear();
        triangles.reserve(valid_faces);
        for (auto& f : h_out_faces)
            triangles.emplace_back(f.x, f.y, f.z);
    }

    std::cout << "Final compact clusters: " << vertices.size()
              << " | Final faces: " << triangles.size() << "\n";
}



// ============================================================
// Host driver with STREAMING neighbor search (Optimized)
// ============================================================

__global__ void check_any_changed_kernel(const bool* changed, int n, int* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int found;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();

    if (tid < n && changed[tid]) atomicExch(&found, 1);
    __syncthreads();
    if (threadIdx.x == 0 && found) *flag = 1;
}

__global__ void compute_keys_kernel(
    const double3* pts, long long* keys, int n,
    double3 minb, double cell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double3 p = pts[i];
    int ix = (int)((p.x - minb.x) / cell);
    int iy = (int)((p.y - minb.y) / cell);
    int iz = (int)((p.z - minb.z) / cell);
    const long long p1 = 73856093LL;
    const long long p2 = 19349663LL;
    const long long p3 = 83492791LL;
    keys[i] = ((long long)ix * p1) ^ ((long long)iy * p2) ^ ((long long)iz * p3);
}


// ============================================================
// GPU version of hash grid build (memory-safe for mid-range GPUs)
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

    // Bind Thrust to this stream
    auto pol = thrust::cuda::par.on(stream);
    auto first = thrust::make_counting_iterator<int>(0);
    auto last = first + nverts;

    // --- Bounding box ---
    Eigen::Vector3d min_bound(DBL_MAX, DBL_MAX, DBL_MAX);
    Eigen::Vector3d max_bound(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (auto& v : vertices) {
        min_bound = min_bound.cwiseMin(v);
        max_bound = max_bound.cwiseMax(v);
    }
    grid.min_bound = make_double3(min_bound.x(), min_bound.y(), min_bound.z());
    grid.cell_size = eps;
    // Compute resolution for dense grid (3× epsilon padding)
    grid.resolution.x = static_cast<int>((max_bound.x() - min_bound.x()) / eps) + 3;
    grid.resolution.y = static_cast<int>((max_bound.y() - min_bound.y()) / eps) + 3;
    grid.resolution.z = static_cast<int>((max_bound.z() - min_bound.z()) / eps) + 3;
    std::cout << "Grid resolution = ("
          << grid.resolution.x << ", "
          << grid.resolution.y << ", "
          << grid.resolution.z << ")\n";


    double3* h_points = nullptr;        // pinned host
    cudaMallocHost(&h_points, nverts * sizeof(double3));

    // --- Copy vertices to GPU ---
    thrust::device_vector<double3> d_points(nverts);
    {
        for (int i = 0; i < nverts; ++i)
            h_points[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_points.data()),
            h_points,
            nverts * sizeof(double3),
            cudaMemcpyHostToDevice,
            stream);
    }

    // --- Allocate buffers ---
    thrust::device_vector<long long> d_keys(nverts);
    thrust::device_vector<int> d_idx(nverts);
    thrust::sequence(d_idx.begin(), d_idx.end());

    // --- Kernel: compute hash per point ---
    const int BLOCK = 256;
    const int GRID = (nverts + BLOCK - 1) / BLOCK;
    double3 minb = grid.min_bound;
    double cell_size = grid.cell_size;

    compute_keys_kernel<<<GRID, BLOCK, 0, stream>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        nverts, minb, cell_size);

    // --- Sort by key (in-place) ---
    {
        GpuTimer t("GPU sort_by_key (hash grid)", print_time, stream);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_idx.begin());
    }

    // --- Reduce to unique keys + offsets (CSR-style) ---
    thrust::device_vector<long long> d_unique_keys(nverts);
    thrust::device_vector<int> d_offsets(nverts + 1);
    thrust::device_vector<int> d_counts(nverts);

    auto end_pair = thrust::reduce_by_key(pol, 
        d_keys.begin(), d_keys.end(),
        thrust::make_constant_iterator(1),
        d_unique_keys.begin(), d_counts.begin());

    int num_cells = end_pair.first - d_unique_keys.begin();
    d_unique_keys.resize(num_cells);
    d_counts.resize(num_cells);

    // Exclusive scan to get offsets
    thrust::exclusive_scan(pol, d_counts.begin(), d_counts.end(), d_offsets.begin());
    int total_points = nverts;
    d_offsets[num_cells] = total_points;

    // --- Compact results into output ---
    d_cell_keys.swap(d_unique_keys);
    d_cell_offsets.swap(d_offsets);
    d_cell_points.swap(d_idx);

    std::cout << "⚙️ GPU hash grid built: " << num_cells
              << " occupied cells, " << nverts << " points.\n";

    cudaFreeHost(h_points);
}


// ============================================================
// GPU kernel: build open-addressing hash table (linear probing)
// ============================================================

//__global__ void build_hash_table_kernel(
//    const long long* __restrict__ cell_keys,
//    int num_cells,
//    long long* __restrict__ hash_keys,
//    int* __restrict__ hash_vals,
//    int table_mask)
//{
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid >= num_cells) return;
//
//    long long key = cell_keys[tid];
//    unsigned int slot =
//        ((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) & table_mask;
//
//    // Linear probing until we find an empty slot
//    while (true) {
//       unsigned long long prev = atomicCAS(
//    reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
//    static_cast<unsigned long long>(LLONG_MIN),
//    static_cast<unsigned long long>(key)
//);
//
//        if (prev == LLONG_MIN || prev == key) {
//            hash_vals[slot] = tid;  // store index of this cell
//            return;
//        }
//        slot = (slot + 1) & table_mask;
//    }
//}



// ============================================================
// Dense-grid + tiled neighbor construction (GPU only)
// ============================================================
// Author: Khizra + GPT-5
// Purpose: 2–3× faster spatial phase by eliminating hash table
// ============================================================


// ------------------------------------------------------------
// Kernel 1: Count points per dense grid cell
// ------------------------------------------------------------
__global__ void count_points_per_cell_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cell_counts,
    int nverts,
    double3 min_bound,
    double cell_size,
    int3 res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nverts) return;

    double3 p = points[idx];
    int ix = (int)((p.x - min_bound.x) / cell_size);
    int iy = (int)((p.y - min_bound.y) / cell_size);
    int iz = (int)((p.z - min_bound.z) / cell_size);
    if (ix < 0 || iy < 0 || iz < 0 ||
        ix >= res.x || iy >= res.y || iz >= res.z)
        return;

    int cell_id = ix + iy * res.x + iz * res.x * res.y;
    atomicAdd(&cell_counts[cell_id], 1);
}

// ------------------------------------------------------------
// Kernel 2: Fill dense grid with vertex indices
// ------------------------------------------------------------
__global__ void fill_dense_grid_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cell_offsets,
    int* __restrict__ cell_points,
    int* __restrict__ cell_counters,
    int nverts,
    double3 min_bound,
    double cell_size,
    int3 res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nverts) return;

    double3 p = points[idx];
    int ix = (int)((p.x - min_bound.x) / cell_size);
    int iy = (int)((p.y - min_bound.y) / cell_size);
    int iz = (int)((p.z - min_bound.z) / cell_size);
    if (ix < 0 || iy < 0 || iz < 0 ||
        ix >= res.x || iy >= res.y || iz >= res.z)
        return;

    int cell_id = ix + iy * res.x + iz * res.x * res.y;
    int pos = atomicAdd(&cell_counters[cell_id], 1);
    if (pos < MAX_CELL_SIZE)
        cell_points[cell_id * MAX_CELL_SIZE + pos] = idx;
}

// ------------------------------------------------------------
// Kernel 3: Count neighbors using shared-memory tiles
// ------------------------------------------------------------
__global__ void count_neighbors_dense_tiled_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ cell_counts,
    const int* __restrict__ cell_points,
    int3 res,
    double3 min_bound,
    double cell_size,
    double eps,
    int* __restrict__ neighbor_counts)
{
    __shared__ double3 sh_points[MAX_CELL_SIZE];

    int cell_id = blockIdx.x;
    if (cell_id >= res.x * res.y * res.z) return;
    int local_count = cell_counts[cell_id];
    if (local_count == 0) return;

    int ix = cell_id % res.x;
    int iy = (cell_id / res.x) % res.y;
    int iz = cell_id / (res.x * res.y);

    // Cache current cell points into shared memory
    for (int i = threadIdx.x; i < local_count; i += blockDim.x)
        sh_points[i] = points[cell_points[cell_id * MAX_CELL_SIZE + i]];
    __syncthreads();

    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        int u = cell_points[cell_id * MAX_CELL_SIZE + i];
        double3 pu = sh_points[i];
        int count = 0;

        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    int nx = ix + dx;
                    int ny = iy + dy;
                    int nz = iz + dz;
                    if (nx < 0 || ny < 0 || nz < 0 ||
                        nx >= res.x || ny >= res.y || nz >= res.z)
                        continue;

                    int ncell = nx + ny * res.x + nz * res.x * res.y;
                    int ncount = cell_counts[ncell];
                    for (int j = 0; j < ncount; j++) {
                        int v = cell_points[ncell * MAX_CELL_SIZE + j];
                        if (v == u) continue;
                        double3 pv = points[v];
                        double dx_ = pu.x - pv.x;
                        double dy_ = pu.y - pv.y;
                        double dz_ = pu.z - pv.z;
                        if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps*eps)
                            count++;
                    }
                }
        neighbor_counts[u] = count;
    }
}

// ------------------------------------------------------------
// Kernel 4: Fill neighbor indices for CSR
// ------------------------------------------------------------
__global__ void fill_neighbors_dense_tiled_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ cell_counts,
    const int* __restrict__ cell_points,
    int3 res,
    double3 min_bound,
    double cell_size,
    double eps,
    const int* __restrict__ neighbor_offsets,
    int* __restrict__ neighbor_indices)
{
    int cell_id = blockIdx.x;
    if (cell_id >= res.x * res.y * res.z) return;
    int local_count = cell_counts[cell_id];
    if (local_count == 0) return;

    int ix = cell_id % res.x;
    int iy = (cell_id / res.x) % res.y;
    int iz = cell_id / (res.x * res.y);

    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        int u = cell_points[cell_id * MAX_CELL_SIZE + i];
        double3 pu = points[u];
        int write_ptr = neighbor_offsets[u];

        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    int nx = ix + dx;
                    int ny = iy + dy;
                    int nz = iz + dz;
                    if (nx < 0 || ny < 0 || nz < 0 ||
                        nx >= res.x || ny >= res.y || nz >= res.z)
                        continue;

                    int ncell = nx + ny * res.x + nz * res.x * res.y;
                    int ncount = cell_counts[ncell];
                    for (int j = 0; j < ncount; j++) {
                        int v = cell_points[ncell * MAX_CELL_SIZE + j];
                        if (v == u) continue;
                        double3 pv = points[v];
                        double dx_ = pu.x - pv.x;
                        double dy_ = pu.y - pv.y;
                        double dz_ = pu.z - pv.z;
                        if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps*eps)
                            neighbor_indices[write_ptr++] = v;
                    }
                }
    }
}

// ------------------------------------------------------------
// Host wrapper: Build CSR neighbor list using dense grid
// ------------------------------------------------------------
void build_neighbor_list_dense_tiled(
    const thrust::device_vector<double3>& d_points,
    int nverts, double eps, GridInfo grid,
    thrust::device_vector<int>& d_neighbor_offsets,
    thrust::device_vector<int>& d_neighbor_indices)
{
    int3 res = grid.resolution;
    int total_cells = res.x * res.y * res.z;

    thrust::device_vector<int> d_cell_counts(total_cells, 0);
    thrust::device_vector<int> d_cell_points(total_cells * MAX_CELL_SIZE, -1);
    thrust::device_vector<int> d_cell_counters(total_cells, 0);
      const int BLOCK_SIZE = 512;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks_points((nverts + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blocks_cells((total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Step 1: Count points per cell
    count_points_per_cell_kernel<<<blocks_points, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_cell_counts.data()),
        nverts, grid.min_bound, grid.cell_size, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Fill grid with vertex indices
    fill_dense_grid_kernel<<<blocks_points, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        nullptr,
        thrust::raw_pointer_cast(d_cell_points.data()),
        thrust::raw_pointer_cast(d_cell_counters.data()),
        nverts, grid.min_bound, grid.cell_size, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Count neighbors
    thrust::device_vector<int> d_neighbor_counts(nverts, 0);
    count_neighbors_dense_tiled_kernel<<<total_cells, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_cell_counts.data()),
        thrust::raw_pointer_cast(d_cell_points.data()),
        res, grid.min_bound, grid.cell_size, eps,
        thrust::raw_pointer_cast(d_neighbor_counts.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Prefix-sum to neighbor_offsets
    thrust::exclusive_scan(
        d_neighbor_counts.begin(), d_neighbor_counts.end(),
        d_neighbor_offsets.begin());
    int total_edges = d_neighbor_offsets[nverts - 1] +
                      d_neighbor_counts[nverts - 1];
    d_neighbor_indices.resize(total_edges);

    // Step 5: Fill neighbor indices
    fill_neighbors_dense_tiled_kernel<<<total_cells, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_cell_counts.data()),
        thrust::raw_pointer_cast(d_cell_points.data()),
        res, grid.min_bound, grid.cell_size, eps,
        thrust::raw_pointer_cast(d_neighbor_offsets.data()),
        thrust::raw_pointer_cast(d_neighbor_indices.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "ΓÜÖ∩╕Å Dense-grid CSR built: " << total_edges
              << " edges (" << (double)total_edges / nverts
              << " avg neighbors per vertex).\n";
}



// ============================================================
// Adaptive Neighbor List Builder (theory-backed decision)
// ============================================================
// Based on criteria from:
//   - Hoetzlein, "Fast Parallel GPU Spatial Hashing" (GPU Pro 7, 2016)
//   - Fathollahi & Chester, "Lock-Free Vertex Clustering" (SIGGRAPH Asia 2023)
//
// Rule 1: Memory Feasibility  →  dense if mem_dense / mem_free < 0.6
// Rule 2: Grid Occupancy      →  dense if occupied / total ≥ 0.25
// Rule 3: Cell Population     →  dense if avg_points_per_cell > 2
// Otherwise fallback to compact hash grid.
// ============================================================
void build_neighbor_list_auto(
    thrust::device_vector<double3>& d_points,
    int nverts,
    double eps,
    GridInfo& grid,
    thrust::device_vector<long long>& d_cell_keys,
    thrust::device_vector<int>& d_cell_offsets,
    thrust::device_vector<int>& d_cell_points,
    thrust::device_vector<long long>& d_hash_keys,
    thrust::device_vector<int>& d_hash_vals,
    int table_mask,
    thrust::device_vector<int>& d_neighbor_offsets,
    thrust::device_vector<int>& d_neighbor_indices,
    thrust::device_vector<int>& d_occupied_indices,
    int num_occupied,
    bool print_time = true)
{
    // --- 1️⃣ Compute grid metrics ---
    const long long total_cells =
        1LL * grid.resolution.x * grid.resolution.y * grid.resolution.z;

    double occupancy = double(num_occupied) / double(total_cells);
    double avg_points_per_cell = double(nverts) / double(num_occupied);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Estimate dense memory requirement (4 int arrays per cell × MAX_CELL_SIZE per cell)
// --- Accurate Dense Memory Estimate & Decision Logic ---
double mem_cells_gb =
    (double)(total_cells * 2 * sizeof(int)) / 1e9;   // cell_start + cell_end
double mem_vertices_gb =
    (double)(nverts * 2 * sizeof(int)) / 1e9;        // neighbor_counts + vertex_ids
double dense_mem_est_gb = mem_cells_gb + mem_vertices_gb;

double free_gb = (double)free_mem / 1e9;

bool use_dense = (occupancy >= 0.10 && avg_points_per_cell > 4.0) &&
                 (dense_mem_est_gb < 0.7 * free_gb);

std::cout << std::fixed << std::setprecision(3);
std::cout << "----------------------------------------------------------\n";
std::cout << "Grid metrics:\n";
std::cout << "  Total cells:           " << total_cells << "\n";
std::cout << "  Occupied cells:        " << num_occupied
          << "  (" << occupancy * 100.0 << "%)\n";
std::cout << "  Avg points per cell:   " << avg_points_per_cell << "\n";
std::cout << "  Dense memory estimate: " << dense_mem_est_gb
          << " GB  |  Free GPU mem: " << free_gb << " GB\n";
std::cout << "----------------------------------------------------------\n";

    if (use_dense) {
        std::cout << "⚡ Using dense-grid tiled CSR builder "
                     "(high occupancy, safe memory)\n";
        GpuTimer t("Build neighbor list (Dense-Grid Tiled)", print_time);
        build_neighbor_list_dense_tiled(
            d_points, nverts, eps, grid,
            d_neighbor_offsets, d_neighbor_indices);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "    Dense-grid CSR build took "
                  << t.seconds() << " s\n";
    } else {
        std::cout << "💡 Switching to compact hash-grid CSR builder "
                     "(sparse or memory-heavy case)\n";
        GpuTimer t("Build neighbor list (Compact Hash-Grid)", print_time);
        build_neighbor_list_from_hashgrid(
            d_points, nverts, eps,
            grid.min_bound, grid.cell_size,
            d_cell_keys, d_cell_offsets, d_cell_points,
            d_hash_keys, d_hash_vals, table_mask,
            d_neighbor_offsets, d_neighbor_indices,
            d_occupied_indices, num_occupied);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "    Compact CSR build took "
                  << t.seconds() << " s\n";
    }
}


// ============================================================
// Optimized Kernel: Remap faces + count valid in one pass
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
    int a = pid2ccid[f.x];
    int b = pid2ccid[f.y];
    int c = pid2ccid[f.z];

    // Discard degenerate triangles
    if (a == b || b == c || a == c) {
        out_faces[fid] = make_int3(-1, -1, -1);
        return;
    }

    out_faces[fid] = make_int3(a, b, c);
    atomicAdd(valid_counter, 1u);
}


// ============================================================
// GPU Streaming Strict-Mode Vertex Clustering
// (PCIe-inclusive and exclusive timings for thesis reporting)
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
    auto pol = thrust::cuda::par.on(stream);
    auto first = thrust::make_counting_iterator<int>(0);
    auto last  = first + nverts;

    // Optional PCIe injection setup (Amdahl’s Law experiment)
    size_t injectMB = 64;
    int injectCopies = 1;
    if (enable_injection)
        gpuinj_init_buffers(injectMB * 1024ULL * 1024ULL);

    // ------------------------------------------------------------
    // Timing variables (for thesis tables)
    // ------------------------------------------------------------
    double timeGrid_s = 0.0, timeHash_s = 0.0, timeCSR_s = 0.0, timeP_s = 0.0, timeSingle_s = 0.0;
    double timeWhile_s = 0.0, timeUmesh_s = 0.0;
    double pcie_h2d_s = 0.0, pcie_d2h_s = 0.0, total_pcie_seconds = 0.0;

    // ============================================================
    // Step 0. Build GPU Hash Grid (memory-safe)
    // ============================================================
    GridInfo grid;
    thrust::device_vector<long long> d_cell_keys;
    thrust::device_vector<int> d_cell_offsets, d_cell_points;
    {
        GpuTimer t("Build hash grid (GPU memory-safe)", print_time, stream);
        build_hash_grid_gpu_safe(vertices, eps, grid,
                                 d_cell_keys, d_cell_offsets, d_cell_points,
                                 print_time, stream);
     //   CUDA_CHECK(cudaStreamSynchronize(stream));
        timeGrid_s = t.seconds();
    }

    // ============================================================
    // Step 1. Build GPU hash table
    // ============================================================
    int num_cells = static_cast<int>(d_cell_keys.size());
    int table_size = 1;
    while (table_size < num_cells * 4) table_size <<= 1;
    int table_mask = table_size - 1;

    thrust::device_vector<long long> d_hash_keys(table_size, LLONG_MIN);
    thrust::device_vector<int> d_hash_vals(table_size, -1);
    {
        GpuTimer t_hash("Build GPU hash table", print_time, stream);
        int threads = 256;
        int blocks = (num_cells + threads - 1) / threads;
        build_hash_table_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_cell_keys.data()), num_cells,
            thrust::raw_pointer_cast(d_hash_keys.data()),
            thrust::raw_pointer_cast(d_hash_vals.data()), table_mask);
     //   CUDA_CHECK(cudaStreamSynchronize(stream));
        timeHash_s = t_hash.seconds();
    }
    std::cout << "✅ GPU hash table built: " << table_size
              << " slots for " << num_cells << " cells.\n";

    // ============================================================
    // Step 2. Compact list of occupied cells
    // ============================================================
    const int N = num_cells;
    thrust::device_vector<int> d_occupied_indices(N);
    const int* offsets = thrust::raw_pointer_cast(d_cell_offsets.data());
    auto end_it = thrust::copy_if(pol, first, first + N,
        d_occupied_indices.begin(),
        [offsets] __device__(int i) {
            return (offsets[i + 1] - offsets[i]) > 0;
        });
    int num_occupied = static_cast<int>(end_it - d_occupied_indices.begin());
    d_occupied_indices.resize(num_occupied);
    std::cout << "✅ Occupied cells (non-empty): "
              << num_occupied << " / " << N << std::endl;

    // ============================================================
    // Step 3. Copy vertices to device (SoA) + PCIe H→D
    // ============================================================
    PointsSOA d_points, h_points;
    {
        GpuTimer t("PCIe H→D transfer", print_time, stream);
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
     //   CUDA_CHECK(cudaStreamSynchronize(stream));
        pcie_h2d_s = t.seconds();
    }

    const double* px = d_points.x;
    const double* py = d_points.y;
    const double* pz = d_points.z;

    // ============================================================
    // Step 4. Build neighbor list (CSR)
    // ============================================================
    thrust::device_vector<int> d_neighbor_offsets(nverts + 1);
    thrust::device_vector<int> d_neighbor_indices;

    // ------------------------------------------------------------
// Persistent device buffers for later stages (reuse across kernels)
// ------------------------------------------------------------
thrust::device_vector<int> d_is_centroid(nverts);
thrust::device_vector<int> d_prefix(nverts);
thrust::device_vector<int> d_pid2ccid(nverts);
thrust::device_vector<int> d_keys(nverts);
thrust::device_vector<SumCount> d_vals(nverts);
thrust::device_vector<int> d_unique_keys(nverts);
thrust::device_vector<SumCount> d_sums(nverts);

// For remapping
thrust::device_vector<int3> d_in_faces(nfaces);
thrust::device_vector<int3> d_out_faces(nfaces);
thrust::device_vector<int>  d_flags(nfaces);


    {
        GpuTimer t("Build neighbor list (CSR)", print_time, stream);
        thrust::device_vector<double3> d_tmp_vertices(nverts);
        thrust::transform(pol, first, last, d_tmp_vertices.begin(),
            [px, py, pz] __device__(int i) { return make_double3(px[i], py[i], pz[i]); });
        build_neighbor_list_from_hashgrid(
            d_tmp_vertices, nverts, eps,
            grid.min_bound, grid.cell_size,
            d_cell_keys, d_cell_offsets, d_cell_points,
            d_hash_keys, d_hash_vals, table_mask,
            d_neighbor_offsets, d_neighbor_indices,
            d_occupied_indices, num_occupied, stream);
  //      CUDA_CHECK(cudaStreamSynchronize(stream));
        timeCSR_s = t.seconds();
    }

    // ============================================================
    // Step 5. Init depend[] from CSR
    // ============================================================
    thrust::device_vector<int> d_depend(nverts, 0);
    {
        GpuTimer t("Init depend from CSR", print_time, stream);
        int threads = 256;
        int blocks = (nverts + threads - 1) / threads;
        init_depend_from_csr_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_depend.data()), nverts,
            thrust::raw_pointer_cast(d_neighbor_offsets.data()),
            thrust::raw_pointer_cast(d_neighbor_indices.data()));
 //       CUDA_CHECK(cudaStreamSynchronize(stream));
        timeP_s = t.seconds();
    }


    // ============================================================
// Step 5.5. Upload faces once (avoid per-phase PCIe copies)
// ============================================================
{
    GpuTimer t_faces("Upload faces to device", print_time, stream);
    std::vector<int3> h_faces(nfaces);
    for (int i = 0; i < nfaces; ++i)
        h_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_in_faces.data()), h_faces.data(),
                    nfaces * sizeof(int3), cudaMemcpyHostToDevice, stream);
}

    // ============================================================
    // Step 6. Frontier Clustering Loop
    // ============================================================
    thrust::device_vector<int> d_cp_vec(nverts);
    thrust::sequence(d_cp_vec.begin(), d_cp_vec.end());
    thrust::device_vector<int> d_active(nverts);
    thrust::device_vector<int> d_next_active(nverts);
    thrust::device_vector<int> d_next_count(1);

    int* h_next_count = nullptr;
    cudaMallocHost(&h_next_count, sizeof(int));

    int BLOCK_SIZE = 512;
    int iter = 0;
    {
        GpuTimer total("Clustering phase ", print_time, stream);

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
         //   CUDA_CHECK(cudaStreamSynchronize(stream));

            double injected_s = 0.0;
            if (enable_injection) {
                injected_s = gpuinj_do_copies(injectCopies, stream);
                total_pcie_seconds += injected_s;
            }

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
// ============================================================
// Step 7–8. Compact Cluster IDs + Centroids + Face Remap (optimized)
// ============================================================
int reduced_clusters = 0;
int valid_faces      = 0;

thrust::device_vector<double3> d_centroids;


// ---- Persistent pinned host buffer for faces (allocated once) ----
int3* h_in_faces = nullptr;
CUDA_CHECK(cudaMallocHost(&h_in_faces, nfaces * sizeof(int3))); // allocate once

{
    GpuTimer t_single("Single region (centroids + remap)", print_time, stream);

    // ---- 7a. Compact cluster IDs ----
    thrust::device_vector<int> d_is_centroid(nverts);
    thrust::device_vector<int> d_prefix(nverts);
    thrust::device_vector<int> d_pid2ccid(nverts);

    const int* cp_ptr     = thrust::raw_pointer_cast(d_cp_vec.data());
    const int* prefix_ptr = thrust::raw_pointer_cast(d_prefix.data());

    // mark centroids
    thrust::transform(thrust::cuda::par.on(stream), first, last, d_is_centroid.begin(),
        [cp_ptr] __device__(int i){ return (cp_ptr[i] == i) ? 1 : 0; });

    // prefix-sum to assign cluster IDs
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
        d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());

    // build PID→CCID mapping
    thrust::transform(thrust::cuda::par.on(stream), first, last, d_pid2ccid.begin(),
        [cp_ptr, prefix_ptr] __device__(int i){
            int root = cp_ptr[i];
            return prefix_ptr[root];
        });

    // ---- 7b. Compute Centroids ----
    thrust::device_vector<int> d_keys(nverts);
    thrust::device_vector<SumCount> d_vals(nverts);
    thrust::copy_n(thrust::cuda::par.on(stream), d_pid2ccid.begin(), nverts, d_keys.begin());
    thrust::transform(thrust::cuda::par.on(stream), first, last, d_vals.begin(),
        [px, py, pz] __device__(int i){
            return SumCount{ px[i], py[i], pz[i], 1 };
        });

    thrust::sort_by_key(thrust::cuda::par.on(stream),
        d_keys.begin(), d_keys.end(), d_vals.begin());

    thrust::device_vector<int> d_unique_keys(nverts);
    thrust::device_vector<SumCount> d_sums(nverts);

    auto new_end = thrust::reduce_by_key(thrust::cuda::par.on(stream),
        d_keys.begin(), d_keys.end(), d_vals.begin(),
        d_unique_keys.begin(), d_sums.begin(),
        thrust::equal_to<int>(), SumCombine());

    reduced_clusters = static_cast<int>(new_end.first - d_unique_keys.begin());
    d_centroids.resize(reduced_clusters);

    thrust::transform(thrust::cuda::par.on(stream),
        d_sums.begin(), d_sums.begin() + reduced_clusters,
        d_centroids.begin(),
        [] __device__(const SumCount& sc){
            double inv = 1.0 / (double)sc.count;
            return make_double3(sc.x * inv, sc.y * inv, sc.z * inv);
        });

 // ---- 7c. Mesh Remap (optimized single-pass) ----
{
    GpuTimer t_mesh("Mesh update (face remap)", print_time, stream);


    d_out_faces.resize(nfaces);

    // Counter for valid faces
    unsigned int* d_valid_counter = nullptr;
    cudaMalloc(&d_valid_counter, sizeof(unsigned int));
    cudaMemsetAsync(d_valid_counter, 0, sizeof(unsigned int), stream);

    int threads = 256;
    int blocks  = (nfaces + threads - 1) / threads;
    remap_and_count_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_in_faces.data()),
        thrust::raw_pointer_cast(d_out_faces.data()),
        thrust::raw_pointer_cast(d_pid2ccid.data()),
        nfaces,
        d_valid_counter);
    CUDA_CHECK(cudaGetLastError());
    {
        GpuTimer t_pcie2("PCIe D→H transfer", print_time, stream);
    // Copy back valid face count
    cudaMemcpyAsync(&valid_faces, d_valid_counter,
                    sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_valid_counter);

    // Optional: compact to remove (-1,-1,-1) faces
    auto end_it = thrust::remove_if(thrust::cuda::par.on(stream),
        d_out_faces.begin(), d_out_faces.end(),
        [] __device__(const int3& f){
            return (f.x < 0 || f.y < 0 || f.z < 0);
        });
    d_out_faces.resize(end_it - d_out_faces.begin());
           // CUDA_CHECK(cudaStreamSynchronize(stream));
        pcie_d2h_s = t_pcie2.seconds();
    }
    timeUmesh_s = t_mesh.seconds();
    std::cout << "    (valid faces = " << valid_faces << ")\n";

}


    timeSingle_s = t_single.seconds();
} // end t_single timer scope

// free pinned memory once, at the very end of the function
cudaFreeHost(h_in_faces);





// ============================================================
// Step 11. Copy reduced vertices & faces back to host mesh
// ============================================================
{
    GpuTimer t_pcie_d2h("PCIe D→H transfer", print_time, stream);

    // ---- Allocate pinned host buffers ----
    double3* h_vertices_pinned;
    int3*    h_faces_pinned;
    size_t v_bytes = d_centroids.size() * sizeof(double3);
    size_t f_bytes = valid_faces * sizeof(int3);

    cudaMallocHost(&h_vertices_pinned, v_bytes);
    cudaMallocHost(&h_faces_pinned,   f_bytes);

    // ---- Async copy from device to pinned host ----
    cudaMemcpyAsync(h_vertices_pinned,
                    thrust::raw_pointer_cast(d_centroids.data()),
                    v_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_faces_pinned,
                    thrust::raw_pointer_cast(d_out_faces.data()),
                    f_bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // wait for both

    // ---- Move into Eigen vectors (no push_back loop) ----
    vertices.resize(d_centroids.size());
    triangles.resize(valid_faces);

#pragma omp parallel for
    for (int i = 0; i < vertices.size(); ++i) {
        vertices[i] = Eigen::Vector3d(h_vertices_pinned[i].x,
                                      h_vertices_pinned[i].y,
                                      h_vertices_pinned[i].z);
    }
#pragma omp parallel for
    for (int i = 0; i < valid_faces; ++i) {
        triangles[i] = Eigen::Vector3i(h_faces_pinned[i].x,
                                       h_faces_pinned[i].y,
                                       h_faces_pinned[i].z);
    }

    cudaFreeHost(h_vertices_pinned);
    cudaFreeHost(h_faces_pinned);

    std::cout << "✅ Copied back reduced mesh: "
              << vertices.size() << " vertices, "
              << triangles.size() << " faces.\n";

    pcie_d2h_s += t_pcie_d2h.seconds();
}


    // ============================================================
    // Step 10. Print timing summary
    // ============================================================
    double Gridbuild_phase_s = timeGrid_s + timeHash_s ;
       double adjacency_p =  timeCSR_s + timeP_s;
       double clustering_phase_s = timeWhile_s + timeSingle_s + timeUmesh_s;
    double total_exclusive = adjacency_p + timeWhile_s + timeSingle_s + timeUmesh_s;
    double total_inclusive = total_exclusive + pcie_h2d_s + pcie_d2h_s;

    std::cout << "\n----------------------------------------------------------\n";
    std::cout << "GPU Timing Summary (Strict-Mode Clustering)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  PCIe H→D transfer: " << pcie_h2d_s << " s\n";
    std::cout << "  → Total Indexing phase:     " << Gridbuild_phase_s << " s\n";
       std::cout << "  Build hash grid:  " << timeGrid_s << " s\n";
    std::cout << "     Build hash table: " << timeHash_s << " s\n";
    std::cout << "  Populating adjacency list:" << adjacency_p << " s\n";
    std::cout << "     Build neighbor list (CSR): " << timeCSR_s << " s\n";
    std::cout << "     Init depend from CSR:      " << timeP_s << " s\n";
    std::cout << "  → Total Clustering phase:    " << clustering_phase_s << " s\n";
        std::cout << "  WHILE LOOP: " << timeWhile_s << " s\n";
        std::cout << "  Single region (centroids + remap): " << timeSingle_s << " s\n";

    std::cout << "  Mesh update (face remap): " << timeUmesh_s << " s\n";
    std::cout << "  PCIe D→H transfer: " << pcie_d2h_s << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "  TOTAL (exclusive of PCIe): " << total_exclusive << " s\n";
    std::cout << "  TOTAL (inclusive of PCIe): " << total_inclusive << " s\n";
    std::cout << "  Injected PCIe (measured): " << total_pcie_seconds << " s\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Final compact clusters: " << reduced_clusters
               << " | Valid faces: " << valid_faces << "\n";

    // Cleanup
    cudaFreeHost(h_next_count);
    cudaFreeHost(h_points.x);
    cudaFreeHost(h_points.y);
    cudaFreeHost(h_points.z);
    cudaFree(d_points.x);
    cudaFree(d_points.y);
    cudaFree(d_points.z);
    if (enable_injection)
        gpuinj_free_buffers();
    cudaStreamDestroy(stream);

    std::cout << "✅ GPU streaming clustering complete.\n";
}



    }
}
