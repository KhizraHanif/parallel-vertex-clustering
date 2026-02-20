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


#define TILE_X 2
#define TILE_Y 2
#define TILE_Z 2
#define MAX_CELL_SIZE 64   // safe for shared mem < 48KB
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
// 🔹 Parallel hash-grid builder using OpenMP (Option A)
// ============================================================

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
    GpuTimer(const std::string& label, bool print = true)
        : label_(label), print_(print), elapsed_ms_(0.0f)
    {
        if (print_) {
            indent_ = depth_++;
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
            cudaEventRecord(start_);
        }
    }

    // ✅ compute elapsed time immediately (safe before destructor)
    double seconds() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_ms_, start_, stop_);
        return static_cast<double>(elapsed_ms_) / 1000.0;
    }

    ~GpuTimer() {
        if (print_) {
            cudaEventRecord(stop_);
            cudaEventSynchronize(stop_);
            cudaEventElapsedTime(&elapsed_ms_, start_, stop_);
            for (int i = 0; i < indent_; i++)
                std::cout << "    ";
            std::cout << label_ << " took " << elapsed_ms_ / 1000.0 << " seconds\n";
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
            depth_--;
        }
    }

private:
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

__global__ void dummy_kernel() {}


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


        // ============================================================
        // Kernel: Initialize depend[] for streaming clustering
        // ============================================================
 


        __global__ void streaming_init_gpu_depend_kernel(
            const double3* points,
            int* depend,
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
            int u = blockIdx.x * blockDim.x + threadIdx.x;
            if (u >= nverts) return;

            double3 pu = points[u];
            int ix0 = (int)((pu.x - min_bound.x) / cell_size);
            int iy0 = (int)((pu.y - min_bound.y) / cell_size);
            int iz0 = (int)((pu.z - min_bound.z) / cell_size);

            // Search 27 cells
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        long long key = cell_key(ix0 + dx, iy0 + dy, iz0 + dz);
                        int idx = hash_lookup(key, hash_keys, hash_vals, table_mask);

                        if (idx < 0) continue;

                        int start = cell_offsets[idx];
                        int end = cell_offsets[idx + 1];
                        for (int i = start; i < end; i++) {
                            int v = cell_points[i];
                            if (u == v) continue;

                            double dx_ = pu.x - points[v].x;
                            double dy_ = pu.y - points[v].y;
                            double dz_ = pu.z - points[v].z;
                            if (dx_ * dx_ + dy_ * dy_ + dz_ * dz_ <= eps * eps) {
                                if (u < v) {
                                    atomicAdd(&depend[v], 1);
                                }
                            }
                        }
                    }
                }
            }
        }


__global__ void count_neighbors_compact_kernel(
    const double3* __restrict__ points,
    int* __restrict__ neighbor_counts,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    const int* __restrict__ occupied_cells,
    int num_occupied)
{
    __shared__ int shared_cell_offsets[256];
    __shared__ int shared_cell_points[256];

    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_occupied) return;

    int cell_id = occupied_cells[cell_idx];
    int start = cell_offsets[cell_id];
    int end = cell_offsets[cell_id + 1];

    // Load cell offsets and points into shared memory
    if (threadIdx.x < 256) {
        shared_cell_offsets[threadIdx.x] = cell_offsets[threadIdx.x];
        shared_cell_points[threadIdx.x] = cell_points[threadIdx.x];
    }
    __syncthreads();

    // Loop over points in this cell
    for (int a = start; a < end; ++a) {
        int u = cell_points[a];
        double3 pu = points[u];
        // integer grid coord for current cell
        int ix0 = (int)((pu.x - min_bound.x) / cell_size);
        int iy0 = (int)((pu.y - min_bound.y) / cell_size);
        int iz0 = (int)((pu.z - min_bound.z) / cell_size);

        int count = 0;

        // check 27 neighboring cells
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx)
#pragma unroll
            for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
                for (int dz = -1; dz <= 1; ++dz)
                {
                    long long key = cell_key(ix0 + dx, iy0 + dy, iz0 + dz);
                    int idx = binary_search_device(cell_keys, num_occupied, key); // small helper
                    if (idx < 0) continue;

                    int n_start = cell_offsets[idx];
                    int n_end   = cell_offsets[idx + 1];

                    for (int i = n_start; i < n_end; ++i)
                    {
                        int v = cell_points[i];
                        if (v == u) continue;
                        double dx_ = pu.x - points[v].x;
                        double dy_ = pu.y - points[v].y;
                        double dz_ = pu.z - points[v].z;
                        if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps*eps)
                            count++;
                    }
                }
        neighbor_counts[u] = count;
    }
}


// ============================================================
// Step 1b: Fill neighbors after offsets are known
// ============================================================
__global__ void fill_neighbors_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ neighbor_offsets,
    int* __restrict__ neighbor_indices,
    int nverts,
    double eps,
    double3 min_bound,
    double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    int nkeys,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nverts) return;

    double3 pu = points[u];
    int ix0 = (int)((pu.x - min_bound.x) / cell_size);
    int iy0 = (int)((pu.y - min_bound.y) / cell_size);
    int iz0 = (int)((pu.z - min_bound.z) / cell_size);

    int write_ptr = neighbor_offsets[u];

    // Loop over 27 cells again (same as above)
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
                int end   = cell_offsets[idx + 1];

                for (int i = start; i < end; i++) {
                    int v = cell_points[i];
                    if (v == u) continue;
                    double dx_ = pu.x - points[v].x;
                    double dy_ = pu.y - points[v].y;
                    double dz_ = pu.z - points[v].z;
                    if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps * eps)
                        neighbor_indices[write_ptr++] = v;
                }
            }
        }
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
    const int* __restrict__ occupied_cells,  // ✅ only process non-empty cells
    int num_occupied)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= num_occupied) return;

    int cell_id = occupied_cells[cell_idx];
    int start = cell_offsets[cell_id];
    int end   = cell_offsets[cell_id + 1];

    for (int a = start; a < end; ++a)
    {
        int u = cell_points[a];
        double3 pu = points[u];

        int ix0 = (int)((pu.x - min_bound.x) / cell_size);
        int iy0 = (int)((pu.y - min_bound.y) / cell_size);
        int iz0 = (int)((pu.z - min_bound.z) / cell_size);

        // check 27 neighboring cells
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx)
#pragma unroll
            for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
                for (int dz = -1; dz <= 1; ++dz)
                {
                    long long key = cell_key(ix0 + dx, iy0 + dy, iz0 + dz);
                    int idx = binary_search_device(cell_keys, num_occupied, key);
                    if (idx < 0) continue;

                    int n_start = cell_offsets[idx];
                    int n_end   = cell_offsets[idx + 1];

                    for (int i = n_start; i < n_end; ++i)
                    {
                        int v = cell_points[i];
                        if (v == u) continue;

                        double dx_ = pu.x - points[v].x;
                        double dy_ = pu.y - points[v].y;
                        double dz_ = pu.z - points[v].z;
                        if (dx_*dx_ + dy_*dy_ + dz_*dz_ <= eps*eps)
                        {
                            int pos = atomicAdd(&neighbor_counts[u], 1);
                            neighbor_indices[neighbor_offsets[u] + pos] = v;
                        }
                    }
                }
    }
}



// ============================================================
// Build neighbor list (CSR) from hash grid
// ============================================================
// ============================================================
// ⚡ Optimized Neighbor List Construction (Compact Cells + CSR)
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
    thrust::device_vector<int>& d_occupied_indices,   // ✅ added
    int num_occupied)                                 // ✅ added
{
    // ============================================================
    // Step 1: Count neighbors (compact-cell version)
    // ============================================================
    thrust::device_vector<int> d_neighbor_counts(nverts);
    int threads = 256;
    int blocks  = (num_occupied + threads - 1) / threads;

    {
        GpuTimer t("Count neighbors (compact)", true);
        count_neighbors_compact_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_in_vertices.data()),
            thrust::raw_pointer_cast(d_neighbor_counts.data()),
            nverts, eps, min_bound, cell_size,
            thrust::raw_pointer_cast(d_cell_keys.data()),
            thrust::raw_pointer_cast(d_cell_offsets.data()),
            thrust::raw_pointer_cast(d_cell_points.data()),
            thrust::raw_pointer_cast(d_occupied_indices.data()),
            num_occupied);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ============================================================
    // Step 2: Prefix-sum to compute neighbor_offsets (CSR format)
    // ============================================================
    d_neighbor_offsets.resize(nverts);
    thrust::exclusive_scan(
        d_neighbor_counts.begin(),
        d_neighbor_counts.end(),
        d_neighbor_offsets.begin());

    // Compute total neighbors for allocation
    int last_count = 0, last_offset = 0;
    cudaMemcpy(&last_count,
               thrust::raw_pointer_cast(d_neighbor_counts.data() + nverts - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_offset,
               thrust::raw_pointer_cast(d_neighbor_offsets.data() + nverts - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
    int total_neighbors = last_offset + last_count;
    d_neighbor_indices.resize(total_neighbors);

    // ============================================================
    // Step 3: Reset counts for atomic append
    // ============================================================
    thrust::fill(d_neighbor_counts.begin(), d_neighbor_counts.end(), 0);

    // ============================================================
    // Step 4: Fill neighbor list (compact-cell version)
    // ============================================================
    {
        GpuTimer t("Fill neighbors (compact)", true);
        build_neighbors_compact_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_in_vertices.data()),
            thrust::raw_pointer_cast(d_neighbor_offsets.data()),
            thrust::raw_pointer_cast(d_neighbor_counts.data()),
            thrust::raw_pointer_cast(d_neighbor_indices.data()),
            nverts, eps, min_bound, cell_size,
            thrust::raw_pointer_cast(d_cell_keys.data()),
            thrust::raw_pointer_cast(d_cell_offsets.data()),
            thrust::raw_pointer_cast(d_cell_points.data()),
            thrust::raw_pointer_cast(d_occupied_indices.data()),
            num_occupied);
        CUDA_CHECK(cudaDeviceSynchronize());
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
// Deterministic, warp-synchronous strict-mode clustering kernel
// ============================================================
__global__ void strict_frontier_deterministic_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    bool* __restrict__ changed_flags)
{
    // ---- warp mapping ----
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= active_count) return;

    int u = active_vertices[warp_id];
    depend[u] = -1;

    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    int start  = neighbor_offsets[u];
    int end    = neighbor_offsets[u + 1];
    int degree = end - start;

    // ---- deterministic ordered traversal ----
    // (neighbors are already sorted ascending in KDTree adjacency)
    for (int i = lane_id; i < degree; i += 32) {
        int v = neighbor_indices[start + i];
        if (v <= u || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        __syncwarp();

        int old = atomicAdd(&depend[v], -1);
        __syncwarp();

        if (old == 1)
            changed_flags[v] = true;
    }
}


// ============================================================
// Warp-Aggregated Deterministic Strict Clustering Kernel
// ============================================================
__global__ void strict_frontier_deterministic_warpagg_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    bool* __restrict__ changed_flags)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= active_count) return;

    int u = active_vertices[warp_id];
    depend[u] = -1;

    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    int start  = neighbor_offsets[u];
    int end    = neighbor_offsets[u + 1];
    int degree = end - start;

    for (int i = lane_id; i < degree; i += 32) {
        int v = neighbor_indices[start + i];
        if (v <= u || v >= nverts) continue;

        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        __syncwarp();

        // Warp-aggregated atomic decrement:
        int old = atomicAdd(&depend[v], -1);

        // Each thread records locally if depend reached zero
        bool became_zero = (old == 1);

        // Warp ballot tells which threads saw zero
        unsigned mask = __ballot_sync(0xffffffff, became_zero);

        // Each thread safely writes its own flag (no shfl, avoids stall)
        if (became_zero)
            changed_flags[v] = true;
    }
}
// ============================================================
// Deterministic + Cached Frontier Clustering Kernel (Final)
// ============================================================
__global__ void strict_frontier_deterministic_cached_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    bool* __restrict__ changed_flags)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= active_count) return;

    // Each warp handles one vertex
    int u = active_vertices[warp_id];
    if (u < 0 || u >= nverts) return;

    depend[u] = -1;
    int cu = cp_vec[u];
    bool isCentroid = (cu == u);

    // Get neighbor range
    int start = neighbor_offsets[u];
    int end   = neighbor_offsets[u + 1];
    int degree = end - start;

    // Cache a small neighbor batch into registers
    // Each lane loads one neighbor at a stride of 32
    for (int base = 0; base < degree; base += 32) {
        int idx = base + lane_id;
        int v = (idx < degree) ? neighbor_indices[start + idx] : -1;

        // Skip invalids and maintain strict order
        if (v <= u || v >= nverts) continue;

        // Perform deterministic strict-mode operations
        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        __syncwarp();

        int old = atomicAdd(&depend[v], -1);
        bool became_zero = (old == 1);

        // Warp ballot to reduce contention (for stats, not required)
        unsigned mask = __ballot_sync(0xffffffff, became_zero);

        // Deterministically flag next frontier
        if (became_zero)
            changed_flags[v] = true;
    }
}

// ============================================================
// Optimized Deterministic Frontier Kernel (Warp-Aggregated)
// ============================================================
__global__ void strict_frontier_deterministic_warpagg_reduced_kernel(
    const double3* __restrict__ points,
    int* __restrict__ cp_vec,
    int* __restrict__ depend,
    const int* __restrict__ active_vertices,
    int active_count,
    const int* __restrict__ neighbor_indices,
    const int* __restrict__ neighbor_offsets,
    int nverts,
    bool* __restrict__ changed_flags)
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
    int end   = neighbor_offsets[u + 1];
    int degree = end - start;

    for (int base = 0; base < degree; base += 32) {
        int idx = base + lane_id;
        int v = (idx < degree) ? neighbor_indices[start + idx] : -1;
        if (v <= u || v >= nverts) continue;

        // Step 1: propagate centroid
        if (isCentroid && depend[v] > 0)
            atomicMin(&cp_vec[v], cu);

        __syncwarp();

        // Step 2: Warp-aggregated depend decrement
        int mask = __match_any_sync(__activemask(), v);
        int leader = __ffs(mask) - 1;
        int group_size = __popc(mask);
        int old = 0;
        if (lane_id == leader)
            old = atomicAdd(&depend[v], -group_size);
        old = __shfl_sync(mask, old, leader);

        // Step 3: determine which threads see depend[v]==0
        bool became_zero = (old <= group_size && old > 0);
        if (became_zero)
            changed_flags[v] = true;
    }
}
// ============================================================


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
    bool print_time = true)
{
    int nverts = static_cast<int>(vertices.size());
    if (nverts == 0) return;

    // --- Bounding box ---
    Eigen::Vector3d min_bound(DBL_MAX, DBL_MAX, DBL_MAX);
    Eigen::Vector3d max_bound(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (auto& v : vertices) {
        min_bound = min_bound.cwiseMin(v);
        max_bound = max_bound.cwiseMax(v);
    }
    grid.min_bound = make_double3(min_bound.x(), min_bound.y(), min_bound.z());
    grid.cell_size = eps;

    // --- Copy vertices to GPU ---
    thrust::device_vector<double3> d_points(nverts);
    {
        thrust::host_vector<double3> h_points(nverts);
        for (int i = 0; i < nverts; ++i)
            h_points[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());
        d_points = h_points;
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

 compute_keys_kernel<<<GRID, BLOCK>>>(
    thrust::raw_pointer_cast(d_points.data()),
    thrust::raw_pointer_cast(d_keys.data()),
    nverts, minb, cell_size);
CUDA_CHECK(cudaDeviceSynchronize());


    // --- Sort by key (in-place) ---
    {
        GpuTimer t("GPU sort_by_key (hash grid)", print_time);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_idx.begin());
    }

    // --- Reduce to unique keys + offsets (CSR-style) ---
    thrust::device_vector<long long> d_unique_keys(nverts);
    thrust::device_vector<int> d_offsets(nverts + 1);
    thrust::device_vector<int> d_counts(nverts);

    auto end_pair = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        thrust::make_constant_iterator(1),
        d_unique_keys.begin(), d_counts.begin());

    int num_cells = end_pair.first - d_unique_keys.begin();
    d_unique_keys.resize(num_cells);
    d_counts.resize(num_cells);

    // Exclusive scan to get offsets
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
    int total_points = nverts;
    d_offsets[num_cells] = total_points;

    // --- Compact results into output ---
    d_cell_keys.swap(d_unique_keys);
    d_cell_offsets.swap(d_offsets);
    d_cell_points.swap(d_idx);

    std::cout << "⚙️ GPU hash grid built: " << num_cells
              << " occupied cells, " << nverts << " points.\n";
}


// ============================================================
// GPU kernel: build open-addressing hash table (linear probing)
// ============================================================
__global__ void build_hash_table_kernel(
    const long long* __restrict__ cell_keys,
    int num_cells,
    long long* __restrict__ hash_keys,
    int* __restrict__ hash_vals,
    int table_mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_cells) return;

    long long key = cell_keys[tid];
    unsigned int slot =
        ((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) & table_mask;

    // Linear probing until we find an empty slot
    while (true) {
       unsigned long long prev = atomicCAS(
    reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
    static_cast<unsigned long long>(LLONG_MIN),
    static_cast<unsigned long long>(key)
);

        if (prev == LLONG_MIN || prev == key) {
            hash_vals[slot] = tid;  // store index of this cell
            return;
        }
        slot = (slot + 1) & table_mask;
    }
}


// ============================================================
// Main host function
// ============================================================
void merge_vertices_forward_gpu_streaming(
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<Eigen::Vector3i>& triangles,
    double eps,
    bool print_time)
{
    int nverts = (int)vertices.size();
    int nfaces = (int)triangles.size();

    // ============================================================
    // Step 0: Build hash grid (CPU)
    // ============================================================
    std::vector<long long> h_cell_keys;
    std::vector<int> h_cell_offsets, h_cell_points;
    GridInfo grid;
GpuTimer t_grid("Step 0: Build hash grid (GPU memory-safe)", print_time);
thrust::device_vector<long long> d_cell_keys;
thrust::device_vector<int> d_cell_offsets, d_cell_points;
build_hash_grid_gpu_safe(vertices, eps, grid,
                         d_cell_keys, d_cell_offsets, d_cell_points, print_time);
double gpu_grid_time = t_grid.seconds();
std::cout << "GPU hash grid build took " << gpu_grid_time << " seconds\n";


// ============================================================
// 🔹 Step: Build GPU hash table directly on device
// ============================================================
int num_cells = d_cell_keys.size();
int table_size = 1;
while (table_size < num_cells * 4) table_size <<= 1;
int table_mask = table_size - 1;

thrust::device_vector<long long> d_hash_keys(table_size, LLONG_MIN);
thrust::device_vector<int>       d_hash_vals(table_size, -1);

{
    GpuTimer t("Build GPU hash table", print_time);
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    build_hash_table_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_cell_keys.data()),
        num_cells,
        thrust::raw_pointer_cast(d_hash_keys.data()),
        thrust::raw_pointer_cast(d_hash_vals.data()),
        table_mask);
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::cout << "✅ GPU hash table built: " << table_size
          << " slots for " << num_cells << " cells.\n";


// ============================================================
// Step X: Compact list of occupied cells (skip empties)
// ============================================================
   // thrust::device_vector<long long> d_cell_keys = h_cell_keys;
   // thrust::device_vector<int> d_cell_offsets = h_cell_offsets;
  //  thrust::device_vector<int> d_cell_points = h_cell_points;

// ============================================================
// Step X: Compact list of occupied cells (skip empties)
// ============================================================
thrust::device_vector<int> d_occupied_indices(d_cell_keys.size());
thrust::sequence(d_occupied_indices.begin(), d_occupied_indices.end());

// Explicit device execution policy ensures correct capture
{
    thrust::device_ptr<int> offsets_ptr = d_cell_offsets.data();

    auto end_it = thrust::copy_if(
        thrust::device,
        d_occupied_indices.begin(),
        d_occupied_indices.end(),
        d_occupied_indices.begin(),
        [=] __device__(int i) {
            // Keep only cells that contain at least one vertex
            return (offsets_ptr[i + 1] - offsets_ptr[i]) > 0;
        });

    int num_occupied = static_cast<int>(end_it - d_occupied_indices.begin());
    d_occupied_indices.resize(num_occupied);

    std::cout << "✅ Occupied cells (non-empty): " << num_occupied
              << " / " << d_cell_keys.size() << std::endl;
}
    int num_occupied = (int)d_occupied_indices.size();



 //   thrust::device_vector<long long> d_hash_keys = h_hash_keys;
 //   thrust::device_vector<int>       d_hash_vals = h_hash_vals;
//    int table_mask = table_size - 1;

    // ============================================================
    // 🔹 Allocate GPU buffers (SoA layout for vertices)
    // ============================================================
    PointsSOA d_points;
    cudaMalloc(&d_points.x, nverts * sizeof(double));
    cudaMalloc(&d_points.y, nverts * sizeof(double));
    cudaMalloc(&d_points.z, nverts * sizeof(double));

    thrust::device_vector<int> d_cp_vec(nverts);
    thrust::device_vector<int> d_depend(nverts, 0);
    thrust::device_vector<int> d_prefix(nverts);
    thrust::device_vector<int> d_is_centroid(nverts);
    thrust::device_vector<int> d_pid2ccid(nverts);
    thrust::device_vector<int> d_keys(nverts);
    thrust::device_vector<SumCount> d_vals(nverts);

    // ============================================================
    // 🔹 Copy vertices to device (SoA)
    // ============================================================
    std::vector<double> hx(nverts), hy(nverts), hz(nverts);
    for (int i = 0; i < nverts; ++i) {
        hx[i] = vertices[i].x();
        hy[i] = vertices[i].y();
        hz[i] = vertices[i].z();
    }
    cudaMemcpy(d_points.x, hx.data(), nverts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points.y, hy.data(), nverts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points.z, hz.data(), nverts * sizeof(double), cudaMemcpyHostToDevice);

    thrust::sequence(d_cp_vec.begin(), d_cp_vec.end());





    // ============================================================
    // Step 1b: Build reusable neighbor list (CSR)
    // ============================================================
    thrust::device_vector<int> d_neighbor_offsets(nverts + 1);
    thrust::device_vector<int> d_neighbor_indices;
     thrust::device_vector<double3> d_tmp_vertices(nverts);
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nverts),
            d_tmp_vertices.begin(),
            [pts = d_points] __device__(int i) {
                return make_double3(pts.x[i], pts.y[i], pts.z[i]);
            });
  
    double timeP_s = 0.0,  timeCSR_s = 0.0, timeAll_s = 0.0;
{
    GpuTimer t("Build neighbor list (CSR)", print_time);
    build_neighbor_list_from_hashgrid(
        d_tmp_vertices, nverts, eps,
        grid.min_bound, grid.cell_size,
        d_cell_keys, d_cell_offsets, d_cell_points,
        d_hash_keys, d_hash_vals, table_mask,
        d_neighbor_offsets, d_neighbor_indices,
        d_occupied_indices, num_occupied);
    CUDA_CHECK(cudaDeviceSynchronize());
    timeCSR_s = t.seconds();
}

  
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ============================================================
    // Step 2: Initialize depend[] using CSR
    // ============================================================
    {
        GpuTimer t("Init depend from CSR", print_time);
        int threads = 256;
        int blocks = (nverts + threads - 1) / threads;
        init_depend_from_csr_kernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_depend.data()),
            nverts,
            thrust::raw_pointer_cast(d_neighbor_offsets.data()),
            thrust::raw_pointer_cast(d_neighbor_indices.data()));
        cudaStreamSynchronize(stream);
        timeP_s = t.seconds();
    }

   int valid_faces = 0;
int reduced_clusters = 0;  
thrust::device_vector<int3> d_out_faces;
thrust::device_vector<int> d_flags;


    const int BLOCK_SIZE = 512;

    thrust::device_vector<int> d_active(nverts);
    thrust::device_vector<int> d_next_active(nverts);
    thrust::device_vector<int> d_next_count(1);
    {
   
 GpuTimer total("timeAll_s", print_time);
    // ---- initial frontier ----
    auto end_it = thrust::copy_if(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nverts),
        d_depend.begin(), d_active.begin(),
        [] __device__(int dep) { return dep == 0; });
    int active_count = end_it - d_active.begin();

    int iter = 0;
    // ============================================================
// 🔹 Timer for frontier clustering loop (Nima-style "while" time)
// ============================================================

  cudaStream_t stream;
cudaStreamCreate(&stream);

// ...

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


    CUDA_CHECK(cudaPeekAtLastError());

    int h_next_count = 0;
    cudaMemcpyAsync(&h_next_count,
                    thrust::raw_pointer_cast(d_next_count.data()),
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    stream);

    cudaStreamSynchronize(stream);

    if (h_next_count > 0)
        thrust::copy_n(d_next_active.begin(), h_next_count, d_active.begin());

    active_count = h_next_count;
    iter++;
}

cudaStreamDestroy(stream);
    std::cout << "        (iters = " << iter << ")\n";

    // ---- Compact IDs ----
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nverts),
        d_is_centroid.begin(),
        [cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data())] __device__(int i) {
            return (cp_ptr[i] == i) ? 1 : 0;
        });
    thrust::exclusive_scan(d_is_centroid.begin(), d_is_centroid.end(), d_prefix.begin());

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nverts),
        d_pid2ccid.begin(),
        [cp_ptr = thrust::raw_pointer_cast(d_cp_vec.data()),
         prefix_ptr = thrust::raw_pointer_cast(d_prefix.data())] __device__(int i) {
            int root = cp_ptr[i];
            return prefix_ptr[root];
        });

    // ---- Centroid averaging ----
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
        [pts = d_points] __device__(int i) {
            return SumCount{pts.x[i], pts.y[i], pts.z[i], 1};
        });
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    thrust::device_vector<int> d_unique_keys(nverts);
    thrust::device_vector<SumCount> d_sums(nverts);
    auto new_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(), d_vals.begin(),
        d_unique_keys.begin(), d_sums.begin(),
        thrust::equal_to<int>(), SumCombine());
    reduced_clusters = new_end.first - d_unique_keys.begin();  // ✅ assign only
    std::cout << "        (reduced clusters = " << reduced_clusters << ")\n";
   

    // ---- Create new SoA ----
    cudaFree(d_points.x);
    cudaFree(d_points.y);
    cudaFree(d_points.z);
    cudaMalloc(&d_points.x, reduced_clusters * sizeof(double));
    cudaMalloc(&d_points.y, reduced_clusters * sizeof(double));
    cudaMalloc(&d_points.z, reduced_clusters * sizeof(double));

    thrust::for_each_n(
        thrust::make_counting_iterator(0),
        reduced_clusters,
        [pts = d_points,
         sums_ptr = thrust::raw_pointer_cast(d_sums.data())] __device__(int i) {
            pts.x[i] = sums_ptr[i].x / sums_ptr[i].count;
            pts.y[i] = sums_ptr[i].y / sums_ptr[i].count;
            pts.z[i] = sums_ptr[i].z / sums_ptr[i].count;
        });

    timeAll_s = total.seconds();

    }
// ============================================================
// GPU Phase Timing Summary
// ============================================================

double neighbor_build_time = timeCSR_s;   // ⏱ Build neighbor list (CSR)
double depend_init_time    = timeP_s;     // ⏱ Init depend from CSR
double clustering_time     = timeAll_s;   // ⏱ Frontier + centroid (no remap)

// measure face remap separately
double timeU_s = 0.0;
{
    GpuTimer t_mesh("face remap time", print_time);
    // ---- Face remapping ----
    std::vector<int3> h_in_faces(nfaces);
    for (int i = 0; i < nfaces; i++)
        h_in_faces[i] = make_int3(triangles[i].x(), triangles[i].y(), triangles[i].z());
    thrust::device_vector<int3> d_in_faces = h_in_faces;
    d_out_faces.resize(nfaces);
    d_flags.resize(nfaces);

int threads = 256;
int blocks = (nfaces + threads - 1) / threads;
remap_faces_kernel_flags<<<blocks, threads>>>(
    thrust::raw_pointer_cast(d_in_faces.data()),
    thrust::raw_pointer_cast(d_out_faces.data()),
    thrust::raw_pointer_cast(d_flags.data()),
    thrust::raw_pointer_cast(d_pid2ccid.data()),
    nfaces);
    cudaStreamSynchronize(stream);

    valid_faces = thrust::count_if(
        d_flags.begin(), d_flags.end(),
        [] __device__(int f) { return f == 1; });
    std::cout << "    (valid faces = " << valid_faces << ")\n";

    timeU_s = t_mesh.seconds();  // ✅ record mesh-update time
}

// Combined times (Nima-style)
double spatial_phase_time  = neighbor_build_time + depend_init_time;
double total_gpu_time      = spatial_phase_time + clustering_time;  // excludes face remap

// ============================================================
// Print Timing Summary
// ============================================================
std::cout << "\n----------------------------------------------------------\n";
std::cout << "GPU Timing Summary (Strict-Mode Clustering)\n";
std::cout << "----------------------------------------------------------\n";
std::cout << "  Spatial phase (neighbor + depend): "
          << spatial_phase_time << " s\n";
std::cout << "     ├── Build neighbor list (CSR):  "
          << neighbor_build_time << " s\n";
std::cout << "     └── Init depend from CSR:       "
          << depend_init_time << " s\n";
std::cout << "  Clustering phase (frontier + centroid): "
          << clustering_time << " s\n";
std::cout << "  Mesh update (face remap): "
          << timeU_s << " s  [excluded from total]\n";
std::cout << "----------------------------------------------------------\n";
std::cout << "  TOTAL GPU phase (spatial + clustering): "
          << total_gpu_time << " s\n";
std::cout << "----------------------------------------------------------\n";





// ============================================================
// Step 4: Copy back to host
// ============================================================
std::vector<double> hx_out(reduced_clusters), hy_out(reduced_clusters), hz_out(reduced_clusters);
cudaMemcpy(hx_out.data(), d_points.x, reduced_clusters * sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(hy_out.data(), d_points.y, reduced_clusters * sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(hz_out.data(), d_points.z, reduced_clusters * sizeof(double), cudaMemcpyDeviceToHost);

vertices.clear();
vertices.reserve(reduced_clusters);
for (int i = 0; i < reduced_clusters; ++i)
    vertices.emplace_back(hx_out[i], hy_out[i], hz_out[i]);

thrust::host_vector<int3> h_out_faces = d_out_faces;
thrust::host_vector<int> h_flags = d_flags;
triangles.clear();
triangles.reserve(valid_faces);
for (int i = 0; i < h_out_faces.size(); i++)
    if (h_flags[i])
        triangles.emplace_back(h_out_faces[i].x, h_out_faces[i].y, h_out_faces[i].z);

std::cout << "Final compact clusters: " << vertices.size()
          << " | Final faces: " << triangles.size() << "\n";

cudaFree(d_points.x);
cudaFree(d_points.y);
cudaFree(d_points.z);
cudaStreamDestroy(stream);

std::cout << "✅ GPU streaming clustering complete.\n";


}
    }
}