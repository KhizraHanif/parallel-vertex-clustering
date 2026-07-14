// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2023 Nima Fathollahi, Sean Chester
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------
#ifdef USE_CUDA
#include "../include/geometry/PWeldCuda.h"
#endif

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <string>
#include <stdexcept>
#include <functional>
#include "../include/Timer.hpp"
#include <chrono>
#include <omp.h>
#include "../include/Eigen/Dense"
#include "../include/geometry/PWeldCuda.h"
#include "../include/io/TriangleMeshIO.h"
#include "../include/geometry/TriangleMeshPWeld.h"
#include "../include/geometry/KDTreeFlann.h"
enum class Version { OPEN3D = 0, FORWARD, FORWARD_ASYNC, GPU, GPU_STREAMING, GPU_OTF };
std::string programName[] = { "Open3D", "forward", "forward_async", "gpu", "gpu_streaming", "gpu_otf" };



int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Enter the following:\n";
        std::cout << "\t-eps (e.g., 0.001)\n";
        std::cout << "\t-Version:\n";
        std::cout << "\t\t0: Open3D\n\t\t1: forward\n\t\t2: forward_async\n\t\t3: gpu\n";
        std::cout << "\t-path to data (must be .ply)\n";
        std::cout << "\t-number of cores (default: 8)\n";
        std::cout << "\t-output path (.ply)\n";
        std::cout << "\tExample: ./main 0.001 3 ../src/data/xyzrgb_manuscript.ply [4] [../src/data/output.ply]\n";
        return 0;
    }

    const double eps = std::stod(argv[1]);
    const int version = std::stoi(argv[2]);
    const std::string dataPath = argv[3];
    const int numCores = (argc >= 5 ? std::stoi(argv[4]) : 1);
    const std::string outputDir = (argc >= 6 ? argv[5] : "");
    const bool verbose = true;

    std::cout << "Configuration:\n";
    std::cout << "\t-eps: " << eps << "\n";
    std::cout << "\t-program: " << programName[version] << "\n";
    std::cout << "\t-path to dataset: " + dataPath + "\n";
    std::cout << "--**--**--**--**--**--**--**--**--\n";

    auto mesh = open3d::geometry::TriangleMesh();
    open3d::io::ReadTriangleMesh(dataPath, mesh);

    open3d::geometry::KDTreeFlann kdtree;  // declare empty
    if (version <= 3) {                    // only for CPU-based versions
        auto t0 = std::chrono::high_resolution_clock::now();
        kdtree.SetGeometry(mesh);          // build the KD-tree
        auto t1 = std::chrono::high_resolution_clock::now();
        double kd_build_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "KDTree build time: " << kd_build_s << " s\n";
    }


    size_t num_vertices_after_reduction;

    auto mesh_pweld = open3d::geometry::TriangleMeshPWeld(mesh);

    if (verbose) {
        std::cout << "number of original vertices: " << mesh_pweld.vertices_.size() << "\n";
        std::cout << "number of original triangles: " << mesh_pweld.triangles_.size() << "\n";
    }

    omp_set_num_threads(numCores); // set the number of cores for the entire program
    switch ((Version)version)
    {
    case Version::OPEN3D:
        mesh_pweld.MergeCloseVertices(kdtree, eps, true);
        break;
    case Version::FORWARD:
        mesh_pweld.merge_vertices_forward(kdtree, eps, true);
        break;
    case Version::FORWARD_ASYNC:
        mesh_pweld.merge_vertices_forward_async(kdtree, eps, true);
        break;
#ifdef USE_CUDA
    case Version::GPU: {
        std::cout << "? Running GPU clustering (precomputed NN)...\n";
        std::vector<int> neighbor_indices, neighbor_offsets, depend;
        build_strict_adjacency_cpu(
            kdtree,
            mesh_pweld.vertices_,
            eps,
            neighbor_indices,
            neighbor_offsets,
            depend);

        open3d::geometry::merge_vertices_forward_gpu(
            mesh_pweld.vertices_,
            mesh_pweld.triangles_,
            neighbor_indices,
            neighbor_offsets,
            depend,
            eps,
            true);
        break;
    }

    case Version::GPU_STREAMING: {
        std::cout << "? Running GPU streaming clustering...\n";
        bool enable_injection = false;
        if (argc > 6 && strcmp(argv[6], "--inject") == 0)
            enable_injection = true;

        open3d::geometry::merge_vertices_forward_gpu_streaming(
            mesh_pweld.vertices_,
            mesh_pweld.triangles_,
            eps,
            enable_injection,
            true);
        break;
    }

case Version::GPU_OTF: {
        std::cout << "Running GPU On-the-Fly clustering (Model 1)...\n";
        open3d::geometry::merge_vertices_forward_gpu_otf(
            mesh_pweld.vertices_,
            mesh_pweld.triangles_,
            eps,
            true);
        break;
    }

#endif


    }
    // ============================================================
    // Write simplified output (handles both mesh and LiDAR cases)
    // ============================================================
    std::cout << "Preparing reduced output..." << std::endl;

    if (!outputDir.empty()) {
        if (mesh_pweld.triangles_.empty()) {
            // ---- LiDAR point cloud ----
            std::ofstream out(outputDir);
            out << "ply\nformat ascii 1.0\n";
            out << "element vertex " << mesh_pweld.vertices_.size() << "\n";
            out << "property float x\nproperty float y\nproperty float z\n";
            out << "end_header\n";
            for (const auto& v : mesh_pweld.vertices_)
                out << v(0) << " " << v(1) << " " << v(2) << "\n";
            out.close();
            std::cout << "[OK] Wrote simplified point cloud: "
                << mesh_pweld.vertices_.size() << " vertices\n";
        }
        else {
            // ---- Full triangle mesh ----
         open3d::io::WriteTriangleMesh(outputDir, mesh_pweld, false, true);
            std::cout << "[OK] Wrote simplified mesh: "
                << mesh_pweld.vertices_.size() << " vertices\n";
        }
    }
}

