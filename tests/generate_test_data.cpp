// generate_test_data.cpp
// Generates a small test case (Example 1 style) for PD2_LPS_ABC verification.
// Creates a small circular domain with Gaussian pulse IC, no ABCs.
// This allows testing the LPS force computation and time integration.
//
// Usage: ./generate_test_data <jobname>
// Outputs: <jobname>.data1, .data2, .nodes, .bctypes, .bcvalues,
//          .icvalues, .icdvalues, .prenotch, .vinodes

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <jobname>" << std::endl;
    return 0;
  }
  std::string jobname = argv[1];

  // Steel 18Ni1900 material (same as paper)
  const double E    = 190.0 / 1000.0;   // kN/mm^2 = GPa
  const double nu   = 0.25;              // Poisson's ratio (BB-PD limit for verification)
  const double rho  = 8000.0 / 1e18;    // kg/mm^3
  const double G0   = 22170.0 * 1e-9;   // fracture energy J/mm^2

  // Discretization
  const double domain_radius = 30.0;    // mm (small for testing)
  const double dx_ = 0.5;               // grid spacing
  const double dy_ = 0.5;
  const double delta = 2.0;             // horizon = 4*dx
  const double dt_ = 35e-9;             // time step (s)
  const int n_steps = 300;              // 10 output steps

  // Critical stretch (plane strain)
  const double s0_ = std::sqrt(5.0 * M_PI * G0 / (12.0 * E * delta));

  // Generate nodes on uniform grid inside circle
  std::vector<double> nodes_x, nodes_y;
  for (double x = -domain_radius; x <= domain_radius + 1e-10; x += dx_) {
    for (double y = -domain_radius; y <= domain_radius + 1e-10; y += dy_) {
      if (std::sqrt(x * x + y * y) <= domain_radius + 1e-10) {
        nodes_x.push_back(x);
        nodes_y.push_back(y);
      }
    }
  }
  int NodeNo = static_cast<int>(nodes_x.size());
  int StepNo = n_steps;
  int PrenotchNo = 0;
  int VINodesNo = 0;
  int constrainsNo = 0;  // no ABCs for this test
  int maxCloudSize = 0;

  std::cout << "Generated " << NodeNo << " nodes in circle of radius "
            << domain_radius << std::endl;

  // Gaussian pulse IC: u_x = u_y = exp(-r^2/100) for r < 15
  std::vector<double> ic_values(2 * NodeNo, 0.0);
  std::vector<double> icd_values(2 * NodeNo, 0.0);
  for (int i = 0; i < NodeNo; i++) {
    double r = std::sqrt(nodes_x[i] * nodes_x[i] + nodes_y[i] * nodes_y[i]);
    if (r < 15.0) {
      double val = std::exp(-r * r / 100.0);
      ic_values[2 * i]     = val;
      ic_values[2 * i + 1] = val;
    }
  }

  // BC types: all free (-1)
  std::vector<int> bc_types(2 * NodeNo, -1);
  std::vector<double> bc_values(2 * NodeNo, 0.0);

  // Write .data1: NodeNo, StepNo, PrenotchNo, VINodesNo, constrainsNo, maxCloudSize
  {
    std::ofstream f(jobname + ".data1");
    f << NodeNo << std::endl;
    f << StepNo << std::endl;
    f << PrenotchNo << std::endl;
    f << VINodesNo << std::endl;
    f << constrainsNo << std::endl;
    f << maxCloudSize << std::endl;
  }

  // Write .data2: E, Nu, Rho, wf, t0, t1, dt, R, dx, dy, s0
  {
    std::ofstream f(jobname + ".data2");
    f.precision(16);
    f << E << std::endl;
    f << nu << std::endl;
    f << rho << std::endl;
    f << 0.0 << std::endl;         // wf
    f << 0.0 << std::endl;         // t0
    f << StepNo * dt_ << std::endl; // t1
    f << dt_ << std::endl;
    f << delta << std::endl;
    f << dx_ << std::endl;
    f << dy_ << std::endl;
    f << s0_ << std::endl;
  }

  // Write .nodes
  {
    std::ofstream f(jobname + ".nodes");
    f.precision(16);
    for (int i = 0; i < NodeNo; i++) {
      f << nodes_x[i] << " " << nodes_y[i] << std::endl;
    }
  }

  // Write .bctypes
  {
    std::ofstream f(jobname + ".bctypes");
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << bc_types[i] << std::endl;
    }
  }

  // Write .bcvalues
  {
    std::ofstream f(jobname + ".bcvalues");
    f.precision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << bc_values[i] << std::endl;
    }
  }

  // Write .icvalues
  {
    std::ofstream f(jobname + ".icvalues");
    f.precision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << ic_values[i] << std::endl;
    }
  }

  // Write .icdvalues
  {
    std::ofstream f(jobname + ".icdvalues");
    f.precision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << icd_values[i] << std::endl;
    }
  }

  // Write .prenotch (empty)
  {
    std::ofstream f(jobname + ".prenotch");
  }

  // Write .vinodes (empty)
  {
    std::ofstream f(jobname + ".vinodes");
  }

  std::cout << "Test data written to " << jobname << ".*" << std::endl;
  std::cout << "Run: ./PD2_LPS_ABC _test " << jobname << std::endl;

  return 0;
}
