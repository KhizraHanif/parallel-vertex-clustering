// ----------------------------------------------------------------------------
// MIT License
// Copyright (c) 2025
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace open3d {
    namespace geometry {

        /// GPU port of Nima's forward clustering.
        /// Uses KDTree on CPU to build adjacency,
        /// then performs strict forward clustering on GPU.
        void merge_vertices_forward_gpu(
            std::vector<Eigen::Vector3d>& vertices,
            std::vector<Eigen::Vector3i>& triangles,
            const std::vector<int>& neighbor_indices,
            const std::vector<int>& neighbor_offsets,
            const std::vector<int>& depend,   // <-- add depend here
            double eps,
            bool print_time);
        /// GPU forward clustering (streaming version).
/// Builds uniform grid on CPU and performs NN search
/// on-the-fly inside the GPU kernel.
        void merge_vertices_forward_gpu_streaming(
            std::vector<Eigen::Vector3d>& vertices,
            std::vector<Eigen::Vector3i>& triangles,
            double eps,
            bool enable_injection,
            bool print_time);

    } // namespace geometry
} // namespace open3d