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

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <string>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <omp.h>
#include "../include/Eigen/Dense"
#include "../include/Timer.hpp"
// nima
#include "../include/io/TriangleMeshIO.h"
#include "../include/geometry/TriangleMeshPWeld.h"
#include "../include/geometry/KDTreeFlann.h"

enum class Version{OPEN3D=0, FORWARD, FORWARD_ASYNC};
std::string programName[] = {"Open3D", "forward", "forward_async"};
int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Enter the following:\n";
        std::cout << "\t-eps (e.g., 0.001)\n";
        std::cout << "\t-Version:\n\t\t0: Open3D, 1: forward, 2: forward_async\n";
        std::cout << "\t-path to data (must be .ply)\n";
        std::cout << "\t-number of cores for all parallel versions (default: 8)\n";
        std::cout << "\t-output path to write the reduced mesh (must end in .ply)\n";
        std::cout << "\t-e.g., ./main 0.001 1 ../src/data/xyzrgb_manuscript.ply 8 ../src/data/output.ply\n";
        return 0;
    }

    const double eps = std::stod(argv[1]);
    const int version = std::stoi(argv[2]);
    const std::string dataPath = argv[3];
    const int numCores = (argc >= 5 ? std::stoi(argv[4]) : 8);
    const std::string outputDir = (argc >= 6 ? argv[5] : "");
    const bool verbose = true;
    const int num_trials = 20;

    std::cout << "Configuration:\n";
    std::cout << "\t-eps: " << eps << "\n";
    std::cout << "\t-program: " << programName[version] << "\n";
    std::cout << "\t-path to dataset: " << dataPath << "\n";
    std::cout << "\t-cores: " << numCores << "\n";
    std::cout << "--**--**--**--**--**--**--**--**--\n";

    // Load original mesh
    open3d::geometry::TriangleMesh original_mesh;
    if (!open3d::io::ReadTriangleMesh(dataPath, original_mesh)) {
        std::cerr << "Failed to load mesh.\n";
        return 1;
    }

    // Construct KDTree once and reuse
    open3d::geometry::KDTreeFlann kdtree(original_mesh);

    omp_set_num_threads(numCores);

    std::vector<double> runtimes;
    runtimes.reserve(num_trials);

    open3d::geometry::TriangleMeshPWeld last_trial_mesh;

    for (int trial = 0; trial < num_trials; ++trial) {
        open3d::geometry::TriangleMeshPWeld mesh_pweld(original_mesh); // fresh copy

        auto start = std::chrono::high_resolution_clock::now();

        switch ((Version)version) {
        case Version::OPEN3D:
            mesh_pweld.MergeCloseVertices(kdtree, eps, true);
            break;
        case Version::FORWARD:
            mesh_pweld.merge_vertices_forward(kdtree, eps, true);
            break;
        case Version::FORWARD_ASYNC:
            mesh_pweld.merge_vertices_forward_async(kdtree, eps, true);
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        runtimes.push_back(duration_ms);
        std::cout << "Trial " << trial + 1 << " runtime: " << duration_ms << " ms\n";

        if (trial == num_trials - 1) {
            last_trial_mesh = std::move(mesh_pweld); // save last mesh for export
        }
    }

    if (!outputDir.empty()) {
        std::cout << "Writing the simplified mesh to: " << outputDir << "\n";
        open3d::io::WriteTriangleMesh(outputDir, last_trial_mesh, false, true);
    }

    std::cout << "Final vertex count: " << last_trial_mesh.vertices_.size() << "\n";
    std::cout << "Final triangle count: " << last_trial_mesh.triangles_.size() << "\n";

    return 0;
}
