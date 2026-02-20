// ----------------------------------------------------------------------------
// MIT License
// (c) 2023 Nima Fathollahi, Sean Chester
// ----------------------------------------------------------------------------

#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <queue>
#include <tuple>
#include <omp.h>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "../include/Eigen/Dense"
#include "../include/geometry/TriangleMeshPWeld.h"
#include "../include/Timer.hpp"


#include <atomic>
#if defined(_MSC_VER)
#define __sync_fetch_and_add(ptr, value) std::atomic_fetch_add(reinterpret_cast<std::atomic<int>*>(ptr), value)
#define __sync_bool_compare_and_swap(ptr, expected, desired) std::atomic_compare_exchange_strong(reinterpret_cast<std::atomic<int>*>(ptr), &expected, desired)
#endif

// Optional helpers for dumping neighbors (unchanged)
void export_neighbors(const std::vector<std::vector<int>>& pid2nnBigger, const std::string& prefix) {
    std::ofstream flat(prefix + "_flat.txt");
    std::ofstream starts(prefix + "_starts.txt");
    int n = (int)pid2nnBigger.size();
    int offset = 0;
    starts << offset << " ";
    for (const auto& nbrs : pid2nnBigger) {
        for (int v : nbrs) flat << v << " ";
        offset += (int)nbrs.size();
        starts << offset << " ";
    }
    flat << std::endl;
    starts << std::endl;
}

namespace open3d {
    namespace geometry {

        static inline void print_vertices(const std::vector<Eigen::Vector3d>& vertices, const std::string& label) {
            std::cout << "\n?? " << label << ":\n";
            for (size_t i = 0; i < vertices.size(); ++i) {
                std::cout << i << ": (" << vertices[i].x() << ", " << vertices[i].y() << ", " << vertices[i].z() << ")\n";
            }
            std::cout << "--------------------------------------\n";
        }

        void TriangleMeshPWeld::reduce(
            std::vector<int>& pid2ccid,
            std::vector<Eigen::Vector3d>& new_vertices,
            const std::vector<int>& cp_vec,
            int num_vertices) const
        {
            int n_clusters = 0; // number of clusters
            std::vector<int> num_cluster_members_vec(num_vertices, 1);
            new_vertices.reserve(num_vertices);

            for (int i = 0; i < num_vertices; i++) {
                if (cp_vec[i] == i) { // centroid
                    pid2ccid[i] = n_clusters++;
                    new_vertices.push_back(vertices_[i]);
                }
                else { // non-centroid
                    const int ccid = pid2ccid[cp_vec[i]];
                    const int prev = num_cluster_members_vec[ccid]++;
                    new_vertices[ccid] = (vertices_[i] + prev * new_vertices[ccid]) / double(prev + 1);
                    pid2ccid[i] = ccid;
                }
            }
        }

        // -----------------------------------------------------------------------------
        // P-Weld (synchronous)
        // -----------------------------------------------------------------------------
        TriangleMeshPWeld& TriangleMeshPWeld::merge_vertices_forward(KDTreeFlann const& kdtree, double eps, bool print_time)
        {
            //  overall timer ("average time" → timeAll)
            Time overall_timer("average time", print_time);

            Time Pweld_timer("P-Weld internal: all but stack unwinding", print_time);

            std::vector<std::vector<int>> pid2nnBigger;
            std::vector<Eigen::Vector3d> new_vertices;
            std::vector<int> cp_vec, pid2ccid, remainingSmallerFinalsVec;

            int numVertices = 0, numIterations = 0;

            {   // Populating adjacency (→ timeP)
                Time adj_list_timer("Populating adj list", print_time);
                numVertices = static_cast<int>(vertices_.size());
                if (print_time) std::cout << "original vertices: " << numVertices << "\n";

                pid2nnBigger = std::vector<std::vector<int>>(numVertices);
                remainingSmallerFinalsVec = std::vector<int>(numVertices);
                cp_vec = std::vector<int>(numVertices);

#pragma omp parallel for
                for (int i = 0; i < numVertices; ++i) {
                    std::vector<double> dists2;
                    int numSmallerNeighbors =
                        kdtree.SearchRadiusSmallerAndBigger(vertices_[i], eps, pid2nnBigger[i], dists2, i);
                    cp_vec[i] = i;
                    remainingSmallerFinalsVec[i] = numSmallerNeighbors - 1; // all smaller neighbors minus itself
                }
            }

            {   // Clustering phase (→ timeC)
                Time clustering_timer("Clustering", print_time);

                {   // While loop (→ timeW)
                    Time while_timer("While loop", print_time);
                    bool should_continue = true;

#pragma omp parallel
                    {
                        while (should_continue) {
#pragma omp barrier
#pragma omp single
                            { ++numIterations; should_continue = false; }

#pragma omp for reduction (||:should_continue)
                            for (int i = 0; i < numVertices; ++i) {
                                if (remainingSmallerFinalsVec[i] < 0) continue;
                                if (remainingSmallerFinalsVec[i] == 0) { // active source
                                    --remainingSmallerFinalsVec[i];
                                    bool isCentroid = (cp_vec[i] == i);
                                    const auto& inner_vec = pid2nnBigger[i];
                                    for (const int bigger : inner_vec) {
                                        if (isCentroid && remainingSmallerFinalsVec[bigger] > 0) {
                                            int expected; int desired = i;
                                            do {
                                                expected = cp_vec[bigger];
                                                if (desired >= expected) break;
                                            } while (!__sync_bool_compare_and_swap(&cp_vec[bigger], expected, desired));
                                        }
                                        if (remainingSmallerFinalsVec[bigger] >= 1) should_continue = true;
                                        __sync_fetch_and_add(&remainingSmallerFinalsVec[bigger], -1);
                                    }
                                }
                            }
                        }
                    }
                }
                if (print_time) std::cout << "numIterations: " << numIterations << "\n";

                // Single region split into UR + UNV (→ timeS contains parent; timeUR/timeUNV are children)
                {
                    Time single_region_timer("Single region", print_time);

                    pid2ccid.clear();
                    pid2ccid.resize(numVertices);
                    new_vertices.clear();
                    new_vertices.reserve(numVertices);

                    // Update representatives (→ timeUR)
                    {
                        Time ur_timer("Update representatives", print_time);
                        int n_clusters = 0;
                        for (int i = 0; i < numVertices; ++i) {
                            if (cp_vec[i] == i) {
                                pid2ccid[i] = n_clusters++;
                                new_vertices.push_back(vertices_[i]);
                            }
                        }
                    }

                    // Update new vertices (→ timeUNV)
                    {
                        Time unv_timer("Update new vertices", print_time);
                        std::vector<int> members(std::max(1, (int)new_vertices.size()), 1);
                        for (int i = 0; i < numVertices; ++i) {
                            if (cp_vec[i] != i) {
                                const int ccid = pid2ccid[cp_vec[i]];
                                const int prev = members[ccid]++;
                                new_vertices[ccid] =
                                    (vertices_[i] + prev * new_vertices[ccid]) / double(prev + 1);
                                pid2ccid[i] = ccid;
                            }
                        }
                    }
                }
            }

            {   // Update mesh (→ timeU) + vertices after
                Time mesh_timer("Update mesh", print_time);
#pragma omp parallel for
                for (auto& tri : triangles_) {
                    tri(0) = pid2ccid[tri(0)];
                    tri(1) = pid2ccid[tri(1)];
                    tri(2) = pid2ccid[tri(2)];
                }
                std::swap(vertices_, new_vertices);
                if (print_time) std::cout << "vertices after: " << vertices_.size() << "\n";
            }

            return *this;
        }

        // -----------------------------------------------------------------------------
        // P-Weld (asynchronous)
        // -----------------------------------------------------------------------------
        TriangleMeshPWeld& TriangleMeshPWeld::merge_vertices_forward_async(KDTreeFlann const& kdtree, double eps, bool print_time)
        {
            //  overall timer ("average time" → timeAll)
            Time overall_timer("average time", print_time);

            // Keep label stable so parsers can share rules if needed
            Time Pweld_async_timer("P-Weld internal: all but stack unwinding", print_time);

            std::vector<std::vector<int>> pid2nnBigger;
            std::vector<Eigen::Vector3d> new_vertices;
            std::vector<int> cp_vec, pid2ccid, remainingSmallerFinalsVec, num_discovered_centroids;

            int numVertices = 0, numIterations = 0, num_threads = 0;
            const int ints_per_cache_line = 16;

            {   // Populating adjacency (→ timeP)
                Time adj_list_timer("Populating adj list", print_time);
                numVertices = static_cast<int>(vertices_.size());
                if (print_time) std::cout << "original vertices: " << numVertices << "\n";

                num_threads = omp_get_max_threads();
                pid2nnBigger = std::vector<std::vector<int>>(numVertices);
                pid2ccid = std::vector<int>(numVertices);
                remainingSmallerFinalsVec = std::vector<int>(numVertices);
                cp_vec = std::vector<int>(numVertices);
                num_discovered_centroids = std::vector<int>(ints_per_cache_line * num_threads + 1, 0);

#pragma omp parallel for
                for (int i = 0; i < numVertices; ++i) {
                    std::vector<double> dists2;
                    int numSmallerNeighbors =
                        kdtree.SearchRadiusSmallerAndBigger(vertices_[i], eps, pid2nnBigger[i], dists2, i);
                    cp_vec[i] = i;
                    remainingSmallerFinalsVec[i] = numSmallerNeighbors - 1;
                }
            }

            {   // Clustering (→ timeC)
                Time clustering_timer("Clustering", print_time);

                {   // While loop (→ timeW)
                    Time while_timer("While loop", print_time);
                    bool should_continue = true;

#pragma omp parallel
                    {
                        while (should_continue) {
#pragma omp barrier
#pragma omp single
                            { ++numIterations; should_continue = false; }

#pragma omp for reduction (||:should_continue)
                            for (int i = 0; i < numVertices; ++i) {
                                if (remainingSmallerFinalsVec[i] < 0) continue;
                                if (remainingSmallerFinalsVec[i] == 0) {
                                    --remainingSmallerFinalsVec[i];
                                    bool isCentroid = (cp_vec[i] == i);
                                    num_discovered_centroids[omp_get_thread_num() * ints_per_cache_line] += (int)isCentroid;
                                    const auto& inner_vec = pid2nnBigger[i];
                                    for (const int bigger : inner_vec) {
                                        if (isCentroid && remainingSmallerFinalsVec[bigger] > 0) {
                                            int expected; int desired = i;
                                            do {
                                                expected = cp_vec[bigger];
                                                if (desired >= expected) break;
                                            } while (!__sync_bool_compare_and_swap(&cp_vec[bigger], expected, desired));
                                        }
                                        if (remainingSmallerFinalsVec[bigger] >= 1) should_continue = true;
                                        __sync_fetch_and_add(&remainingSmallerFinalsVec[bigger], -1);
                                    }
                                }
                            }
                        }
                    }
                }
                if (print_time) std::cout << "numIterations: " << numIterations << "\n";

                {   // Single region (manual) with UR + UNV + normalized mesh label
                    Time single_region_timer("Single region (manual)", print_time);

                    // ---- Update representatives (→ timeUR) ----
                    {
                        Time ur_timer("Update representatives", print_time);
                        std::exclusive_scan(std::cbegin(num_discovered_centroids),
                            std::cend(num_discovered_centroids),
                            std::begin(num_discovered_centroids),
                            0u);
                        new_vertices.resize(num_discovered_centroids[num_threads * ints_per_cache_line]);

#pragma omp parallel
                        {
                            const int th_id = omp_get_thread_num();
                            const int offset = num_discovered_centroids[th_id * ints_per_cache_line];
                            int current_vertex_id = 0;

#pragma omp for
                            for (int i = 0; i < numVertices; ++i) {
                                if (cp_vec[i] == i) { // centroid
                                    const int ccid = offset + current_vertex_id++;
                                    new_vertices[ccid] = vertices_[i];
                                    pid2ccid[i] = ccid;
                                }
                            }
                        }
                    }

                    const int num_clusters = num_discovered_centroids[num_threads * ints_per_cache_line];

                    // ---- Update new vertices (→ timeUNV) ----
                    {
                        Time unv_timer("Update new vertices", print_time);
                        std::vector<unsigned int> num_cluster_members_vec(num_clusters, 1u);

#pragma omp parallel
                        {
#pragma omp single nowait
                            for (int i = 0; i < numVertices; ++i) {
                                if (cp_vec[i] != i) {
                                    const int ccid = pid2ccid[cp_vec[i]];
                                    new_vertices[ccid] =
                                        (new_vertices[ccid] * num_cluster_members_vec[ccid] + vertices_[i]) /
                                        static_cast<float>(num_cluster_members_vec[ccid] + 1);
                                    ++num_cluster_members_vec[ccid];
                                }
                            }
                        }
                    }

                    // ---- Normalize mesh timer label to exactly "Update mesh" (→ timeU) ----
                    {
                        Time mesh_timer("Update mesh", print_time);
#pragma omp for schedule (dynamic, num_clusters / num_threads / 2u)
                        for (auto& tri : triangles_) {
                            tri(0) = pid2ccid[cp_vec[tri(0)]];
                            tri(1) = pid2ccid[cp_vec[tri(1)]];
                            tri(2) = pid2ccid[cp_vec[tri(2)]];
                        }
                    }

                    std::swap(vertices_, new_vertices);
                    if (print_time) std::cout << "vertices after: " << vertices_.size() << "\n";
                }
            }

            return *this;
        }

        void build_strict_adjacency_cpu(
            open3d::geometry::KDTreeFlann& kdtree,
            const std::vector<Eigen::Vector3d>& vertices,
            double eps,
            std::vector<int>& neighbor_indices,
            std::vector<int>& neighbor_offsets,
            std::vector<int>& depend)
        {
            int n = vertices.size();
            neighbor_offsets.resize(n + 1);
            neighbor_offsets[0] = 0;
            depend.resize(n, 0);

            std::vector<int> neighbors;
            std::vector<double> dists;

            // Open file for debugging
            std::ofstream fout("neighbors_kdtree.txt");
            if (!fout) {
                throw std::runtime_error("Cannot open neighbors_kdtree.txt for writing");
            }

            for (int u = 0; u < n; u++) {
                int k = kdtree.SearchRadius(vertices[u], eps, neighbors, dists);

                fout << u;
                for (int v : neighbors) {
                    if (v == u) continue; // skip self
                    if (v < u) {
                        depend[u]++; // smaller neighbor
                    }
                    else {
                        neighbor_indices.push_back(v); // only store bigger ones
                    }
                    fout << " " << v; // dump all neighbors for comparison
                }
                fout << "\n";

                neighbor_offsets[u + 1] = (int)neighbor_indices.size();
            }

            fout.close();
            std::cout << "✅ Neighbor list dumped to neighbors_kdtree.txt\n";
        }
    } // namespace geometry
} // namespace open3d
