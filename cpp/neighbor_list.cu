// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2025 Khizra Hanif
//
// GPU neighbor list construction using sparse hash grid and CSR format.
// Implements a two-pass approach: count neighbors, prefix-sum, fill indices.
// Uses SplitMix64 hashing with open addressing for O(1) cell lookup.
// ----------------------------------------------------------------------------

#include "neighbor_list.cuh"
#include <cfloat>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

using CellKeyT = long long;

// ----------------------------------------------------------------------------
// Spatial hashing
// ----------------------------------------------------------------------------

// Hash a 3D cell coordinate into a 64-bit key using large prime multipliers.
__host__ __device__ inline long long nl_cell_key(int ix, int iy, int iz) {
    const long long p1 = 73856093;
    const long long p2 = 19349663;
    const long long p3 = 83492791;
    return ((long long)ix * p1) ^ ((long long)iy * p2) ^ ((long long)iz * p3);
}

// SplitMix64-style mixing function for open-addressing hash table.
__device__ __forceinline__ unsigned long long nl_mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Insert cell keys into open-addressing hash table using double hashing.
__global__ void nl_build_hash_table_kernel(
    const long long* __restrict__ cell_keys,
    int num_cells,
    long long* __restrict__ hash_keys,
    int* __restrict__ hash_vals,
    long long table_mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_cells) return;

    unsigned long long h    = nl_mix64((unsigned long long)cell_keys[tid]);
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
__device__ __forceinline__ int nl_hash_lookup(
    long long key,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    long long table_mask)
{
    unsigned long long h    = nl_mix64((unsigned long long)key);
    unsigned int       slot = h & table_mask;
    unsigned int       step = ((h >> 32) | 1u);

    while (true) {
        long long stored = hash_keys[slot];
        if (stored == key)       return hash_vals[slot];
        if (stored == LLONG_MIN) return -1;
        slot = (slot + step) & table_mask;
    }
}

// Binary upper-bound search within a sorted cell segment.
__device__ __forceinline__ int nl_upper_bound(const int* a, int len, int val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&a[mid]) <= val) lo = mid + 1;
        else                        hi = mid;
    }
    return lo;
}

// ----------------------------------------------------------------------------
// Hash grid construction
// ----------------------------------------------------------------------------

// Compute per-vertex cell hash keys on device.
__global__ void nl_compute_keys_kernel(
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

// Build sparse hash grid on GPU. Returns cell keys, offsets, and point indices.
static void nl_build_hash_grid(
    const std::vector<Eigen::Vector3d>& vertices,
    double eps,
    double3& out_min_bound,
    double& out_cell_size,
    thrust::device_vector<long long>& d_cell_keys,
    thrust::device_vector<int>& d_cell_offsets,
    thrust::device_vector<int>& d_cell_points)
{
    int nverts = static_cast<int>(vertices.size());

    Eigen::Vector3d min_bound(DBL_MAX,  DBL_MAX,  DBL_MAX);
    Eigen::Vector3d max_bound(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (auto& v : vertices) {
        min_bound = min_bound.cwiseMin(v);
        max_bound = max_bound.cwiseMax(v);
    }

    out_min_bound = make_double3(min_bound.x(), min_bound.y(), min_bound.z());
    out_cell_size = eps;

    // Upload vertices
    double3* h_pts = nullptr;
    cudaMallocHost(&h_pts, nverts * sizeof(double3));
    for (int i = 0; i < nverts; ++i)
        h_pts[i] = make_double3(vertices[i].x(), vertices[i].y(), vertices[i].z());

    thrust::device_vector<double3>  d_pts(nverts);
    cudaMemcpy(thrust::raw_pointer_cast(d_pts.data()),
               h_pts, nverts * sizeof(double3), cudaMemcpyHostToDevice);
    cudaFreeHost(h_pts);

    // Compute cell keys
    thrust::device_vector<CellKeyT> d_keys(nverts);
    thrust::device_vector<int>      d_idx(nverts);
    thrust::sequence(d_idx.begin(), d_idx.end());

    nl_compute_keys_kernel<<<(nverts+255)/256, 256>>>(
        thrust::raw_pointer_cast(d_pts.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        nverts, out_min_bound, out_cell_size);

    thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_idx.begin());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce to unique keys + counts
    thrust::device_vector<CellKeyT> d_unique_keys(nverts);
    thrust::device_vector<int>      d_counts(nverts);
    thrust::device_vector<int>      d_offsets(nverts + 1);

    auto end_pair = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        thrust::make_constant_iterator(1),
        d_unique_keys.begin(), d_counts.begin());

    int num_cells = static_cast<int>(end_pair.first - d_unique_keys.begin());
    d_unique_keys.resize(num_cells);
    d_counts.resize(num_cells);

    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
    d_offsets[num_cells] = nverts;

    d_cell_keys    = d_unique_keys;
    d_cell_offsets = d_offsets;
    d_cell_points  = d_idx;
}

// ----------------------------------------------------------------------------
// Neighbor counting and filling (two-pass CSR construction)
// ----------------------------------------------------------------------------

// Pass 1: count neighbors per vertex using hybrid dense/light cell path.
__global__ void nl_count_neighbors_kernel(
    const double3* __restrict__ points,
    int* __restrict__ neighbor_counts,
    int nverts, double eps,
    double3 min_bound, double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    extern __shared__ int sh[];
    const int    TILE = 256;
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x;
         u < nverts; u += blockDim.x * gridDim.x)
    {
        const double3 pu = points[u];
        int count = 0;
        int cx = __double2int_rd((pu.x - min_bound.x) / cell_size);
        int cy = __double2int_rd((pu.y - min_bound.y) / cell_size);
        int cz = __double2int_rd((pu.z - min_bound.z) / cell_size);

#pragma unroll
        for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            long long key = nl_cell_key(cx+dx, cy+dy, cz+dz);
            int       idx = nl_hash_lookup(key, hash_keys, hash_vals, table_mask);
            if (idx < 0) continue;

            int beg = __ldg(&cell_offsets[idx]);
            int end = __ldg(&cell_offsets[idx + 1]);
            int len = end - beg;
            if (len <= 0) continue;

            int pos = beg + nl_upper_bound(cell_points + beg, len, u);

            if (len > TILE) {
                for (int t = pos; t < end; t += TILE) {
                    int chunk = min(TILE, end - t);
                    for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                        sh[i] = __ldg(&cell_points[t + i]);
                    __syncthreads();
#pragma unroll 4
                    for (int i = 0; i < chunk; ++i) {
                        int v = sh[i];
                        double3 pv = points[v];
                        double d2x = pv.x-pu.x, d2y = pv.y-pu.y, d2z = pv.z-pu.z;
                        count += (d2x*d2x + d2y*d2y + d2z*d2z <= eps2);
                    }
                    __syncthreads();
                }
            } else {
#pragma unroll 4
                for (int k = pos; k < end; ++k) {
                    int v = __ldg(&cell_points[k]);
                    double3 pv = points[v];
                    double d2x = pv.x-pu.x, d2y = pv.y-pu.y, d2z = pv.z-pu.z;
                    count += (d2x*d2x + d2y*d2y + d2z*d2z <= eps2);
                }
            }
        }
        neighbor_counts[u] = count;
    }
}

// Pass 2: fill neighbor indices into CSR storage.
__global__ void nl_fill_neighbors_kernel(
    const double3* __restrict__ points,
    const int* __restrict__ neighbor_offsets,
    int* __restrict__ neighbor_indices,
    int nverts, double eps,
    double3 min_bound, double cell_size,
    const long long* __restrict__ cell_keys,
    const int* __restrict__ cell_offsets,
    const int* __restrict__ cell_points,
    const long long* __restrict__ hash_keys,
    const int* __restrict__ hash_vals,
    int table_mask)
{
    extern __shared__ int sh[];
    const int    TILE = 256;
    const double eps2 = eps * eps;

    for (int u = blockIdx.x * blockDim.x + threadIdx.x;
         u < nverts; u += blockDim.x * gridDim.x)
    {
        const double3 pu  = points[u];
        int           out = neighbor_offsets[u];
        int           end_row = neighbor_offsets[u + 1];

        int cx = __double2int_rd((pu.x - min_bound.x) / cell_size);
        int cy = __double2int_rd((pu.y - min_bound.y) / cell_size);
        int cz = __double2int_rd((pu.z - min_bound.z) / cell_size);

#pragma unroll
        for (int dz = -1; dz <= 1; ++dz)
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            long long key = nl_cell_key(cx+dx, cy+dy, cz+dz);
            int       idx = nl_hash_lookup(key, hash_keys, hash_vals, table_mask);
            if (idx < 0) continue;

            int beg = __ldg(&cell_offsets[idx]);
            int end = __ldg(&cell_offsets[idx + 1]);
            int len = end - beg;
            if (len <= 0) continue;

            int pos = beg + nl_upper_bound(cell_points + beg, len, u);

            if (len > TILE) {
                for (int t = pos; t < end; t += TILE) {
                    int chunk = min(TILE, end - t);
                    for (int i = threadIdx.x; i < chunk; i += blockDim.x)
                        sh[i] = __ldg(&cell_points[t + i]);
                    __syncthreads();
#pragma unroll 4
                    for (int i = 0; i < chunk; ++i) {
                        int v = sh[i];
                        if (v <= u) continue;
                        double3 pv = points[v];
                        double dxf = pv.x-pu.x, dyf = pv.y-pu.y, dzf = pv.z-pu.z;
                        if (dxf*dxf + dyf*dyf + dzf*dzf <= eps2)
                            if (out < end_row) neighbor_indices[out++] = v;
                    }
                    __syncthreads();
                }
            } else {
#pragma unroll 4
                for (int k = pos; k < end; ++k) {
                    int v = __ldg(&cell_points[k]);
                    if (v <= u) continue;
                    double3 pv = points[v];
                    double dxf = pv.x-pu.x, dyf = pv.y-pu.y, dzf = pv.z-pu.z;
                    if (dxf*dxf + dyf*dyf + dzf*dzf <= eps2)
                        if (out < end_row) neighbor_indices[out++] = v;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

GpuNeighborList build_gpu_neighbor_list(
    const std::vector<Eigen::Vector3d>& points,
    double eps)
{
    int nverts = static_cast<int>(points.size());

    // Step 1: Build hash grid
    double3 min_bound; double cell_size;
    thrust::device_vector<long long> d_cell_keys;
    thrust::device_vector<int>       d_cell_offsets, d_cell_points;

    nl_build_hash_grid(points, eps,
                       min_bound, cell_size,
                       d_cell_keys, d_cell_offsets, d_cell_points);

    int num_cells = static_cast<int>(d_cell_keys.size());

    // Step 2: Build hash table
    int table_size = 1;
    while (table_size < num_cells * 4) table_size <<= 1;
    long long table_mask = table_size - 1;

    thrust::device_vector<long long> d_hash_keys(table_size, LLONG_MIN);
    thrust::device_vector<int>       d_hash_vals(table_size, -1);

    nl_build_hash_table_kernel<<<(num_cells+255)/256, 256>>>(
        thrust::raw_pointer_cast(d_cell_keys.data()), num_cells,
        thrust::raw_pointer_cast(d_hash_keys.data()),
        thrust::raw_pointer_cast(d_hash_vals.data()), table_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Upload points
    thrust::device_vector<double3> d_pts(nverts);
    {
        std::vector<double3> h_pts(nverts);
        for (int i = 0; i < nverts; ++i)
            h_pts[i] = make_double3(points[i].x(), points[i].y(), points[i].z());
        cudaMemcpy(thrust::raw_pointer_cast(d_pts.data()),
                   h_pts.data(), nverts * sizeof(double3), cudaMemcpyHostToDevice);
    }

    // Step 4: Count neighbors (pass 1)
    thrust::device_vector<int> d_counts(nverts, 0);
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       nl_count_neighbors_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    nl_count_neighbors_kernel<<<gridSize, blockSize, blockSize*sizeof(int)>>>(
        thrust::raw_pointer_cast(d_pts.data()),
        thrust::raw_pointer_cast(d_counts.data()),
        nverts, eps, min_bound, cell_size,
        thrust::raw_pointer_cast(d_cell_keys.data()),
        thrust::raw_pointer_cast(d_cell_offsets.data()),
        thrust::raw_pointer_cast(d_cell_points.data()),
        thrust::raw_pointer_cast(d_hash_keys.data()),
        thrust::raw_pointer_cast(d_hash_vals.data()),
        table_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Prefix-sum to get CSR offsets
    thrust::device_vector<int> d_offsets(nverts + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());

    int last_count = 0, last_offset = 0;
    cudaMemcpy(&last_count,  thrust::raw_pointer_cast(d_counts.data()  + nverts - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_offsets.data() + nverts - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int total = last_offset + last_count;

    // Step 6: Fill neighbor indices (pass 2)
    thrust::device_vector<int> d_indices(total);
    thrust::fill(d_counts.begin(), d_counts.end(), 0);

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                       nl_fill_neighbors_kernel, 0, 0);
    gridSize = (nverts + blockSize - 1) / blockSize;

    nl_fill_neighbors_kernel<<<gridSize, blockSize, blockSize*sizeof(int)>>>(
        thrust::raw_pointer_cast(d_pts.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        nverts, eps, min_bound, cell_size,
        thrust::raw_pointer_cast(d_cell_keys.data()),
        thrust::raw_pointer_cast(d_cell_offsets.data()),
        thrust::raw_pointer_cast(d_cell_points.data()),
        thrust::raw_pointer_cast(d_hash_keys.data()),
        thrust::raw_pointer_cast(d_hash_vals.data()),
        table_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 7: Copy back to host
    GpuNeighborList result;
    result.offsets.resize(nverts + 1);
    result.indices.resize(total);
    cudaMemcpy(result.offsets.data(),
               thrust::raw_pointer_cast(d_offsets.data()),
               (nverts + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.indices.data(),
               thrust::raw_pointer_cast(d_indices.data()),
               total * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU neighbor list built: " << total << " edges ("
              << std::fixed << std::setprecision(3)
              << (double)total / nverts << " avg/vertex)\n";

    return result;
}
