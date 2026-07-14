// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2025 Khizra Hanif
//
// GPU neighbor list construction using sparse hash grid and CSR format.
// Compatible with any 3D point cloud represented as Eigen::Vector3d.
// Requires CUDA sm_70 or higher (Volta+).
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <Eigen/Dense>
#include <thrust/device_vector.h>

// CSR-format neighbor list result.
// offsets[u] to offsets[u+1] gives the range of neighbors for vertex u.
// All stored neighbors v satisfy v > u and dist(u,v) <= eps.
struct GpuNeighborList {
    std::vector<int> offsets;   // size: nverts + 1
    std::vector<int> indices;   // size: total neighbor pairs
};

// Build a half-symmetric CSR neighbor list entirely on GPU.
// Points: 3D positions as Eigen::Vector3d
// eps:    neighborhood radius
GpuNeighborList build_gpu_neighbor_list(
    const std::vector<Eigen::Vector3d>& points,
    double eps);
