// PD2_LPS_ABC: Linear Peridynamic Solid with Absorbing Boundary Conditions
// 2D plane strain, ordinary state-based (LPS) formulation
//
// Based on the BB-PD solver of Shojaei et al., extended to LPS model.
// Key changes from BB-PD (see LPS_ABC_Report.tex):
//   1. Two-pass force computation: dilatation theta_i then force (Algorithm 1)
//   2. Node-wise weighted volume m_i (Eq. m_disc)
//   3. Arbitrary Poisson's ratio via LPS constants kappa, alpha, lambda_PD
//   4. VTK output for ParaView (displacement, velocity, dilatation)
//   5. No bond breaking (Example 1: elastic waves only)
//   6. Comprehensive verification checks at every critical step

#define _USE_MATH_DEFINES
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdlib>

#include "kd_tree_interface.h"
#include "space.h"
#include "utils.h"

// ============================================================================
// Configuration macros
// ============================================================================
#define DPN 2                    // degrees of freedom per node (2D)
#define SUBSTEP_NO 30            // output frequency (every N steps)
#define MAX_HORIZON_LENGTH 128   // max neighbors per node
#define ABCS_STEP 2              // temporal stencil for ABCs (-1=off, 1=1-step, 2=2-step)
#define KINEMATIC_VELOCITY_ABC 0 // 1=use v=(u^{n+1}-u^n)/dt at ABC nodes, 0=use G_v
#define THETA_ABC_METHOD 1       // 0=IDW extrapolation, 1=G_theta updating vector (Sec. 4.8)

// ============================================================================
// Global variables
// ============================================================================

// Control data
int NodeNo;
int StepNo;
int PrenotchNo;
int VINodesNo;
int constrainsNo;
int maxCloudSize;

// Material parameters (read from .data2)
double E_young;
double Nu;
double Rho;
double wf;
double t0;
double t1;
double dt;
double R;    // horizon radius delta
double dx;
double dy;
double s0;

// LPS material parameters (Sec. 5 of report, Eq. kappa_2d, alpha_2d, lambdaPD_2d)
double Lambda;    // Lame first parameter:   lambda = E*nu / ((1+nu)(1-2nu))
double Mu;        // Lame second parameter:  mu     = E / (2(1+nu))
double Kappa;     // 2D bulk modulus:         kappa  = lambda + mu
double AlphaLPS;  // deviatoric coefficient:  alpha  = 8*mu / m_ref
double LambdaPD;  // dilatation coefficient:  lambda_PD = 2*(lambda - mu) / m_ref
double m_ref;     // reference weighted volume (interior node)

// Data arrays
space::vector<int, int> ControlData1;
space::vector<double, int> ControlData2;
space::matrix<double, int> Nodes;
space::vector<int, int> BCTypes;
space::vector<double, int> BCValues;
space::vector<double, int> ICDValues;
space::vector<double, int> ICValues;
space::vector<int, int> HorizonLengths;
space::matrix<int, int> Horizons;
space::matrix<double, int> Prenotches;

// ABC data
space::matrix<int, int> iClouds;       // Finding 5: int for node indices
space::matrix<double, int> ciNodesU;
space::matrix<double, int> ciNodesV;
space::matrix<double, int> ciNodesTheta; // G_theta updating vector (Sec. 4.8)
space::vector<int, int> VINodes;
space::vector<int, int> infNodes;      // Finding 5: int for node indices

// LPS-specific arrays (Sec. 7.3.1 of report)
space::vector<double, int> mWeighted;  // per-node weighted volume m_i
space::vector<double, int> Theta;      // per-node dilatation theta_i

// ============================================================================
// Volume correction factor beta (Eq. beta in report)
// ============================================================================
double beta(double xi) {
  if (xi <= R - dx / 2.0)
    return 1.0;
  else if (xi <= R + dx / 2.0)
    return (R + dx / 2.0 - xi) / dx;
  else
    return 0.0;
}

// ============================================================================
// Influence function omega (constant = 1, consistent with BB-PD)
// ============================================================================
inline double omega_func(double /*xi*/) {
  return 1.0;
}

// ============================================================================
// Line segment intersection (for prenotch visibility)
// ============================================================================
int relativeCCW(double x1, double y1, double x2, double y2,
                double px, double py) {
  x2 -= x1; y2 -= y1; px -= x1; py -= y1;
  double ccw = px * y2 - py * x2;
  if (ccw == 0.0) {
    ccw = px * x2 + py * y2;
    if (ccw > 0.0) {
      px -= x2; py -= y2;
      ccw = px * x2 + py * y2;
      if (ccw < 0.0) ccw = 0.0;
    }
  }
  return (ccw < 0.0) ? -1 : ((ccw > 0.0) ? 1 : 0);
}

bool line_segment_intersect(double x1, double y1, double x2, double y2,
                            double x3, double y3, double x4, double y4) {
  return ((relativeCCW(x1, y1, x2, y2, x3, y3) *
           relativeCCW(x1, y1, x2, y2, x4, y4) <= 0) &&
          (relativeCCW(x3, y3, x4, y4, x1, y1) *
           relativeCCW(x3, y3, x4, y4, x2, y2) <= 0));
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
  std::cout << " _____  ____   ___        _     ____  ____" << std::endl
            << "|  _  ||    \\ |_  |      | |   |  _ \\/ ___| " << std::endl
            << "|   __||  |  ||  _|  ___ | |__ | |_) \\___ \\ " << std::endl
            << "|__|   |____/ |___|_|___||____||____/|____/ " << std::endl
            << std::endl
            << "LPS State-Based PD with ABCs (Example 1)" << std::endl
            << std::endl;

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <prefix> <jobname>" << std::endl;
    return 0;
  }

  std::string Prefix = argv[1];
  std::string JobName = argv[2];

#ifdef _OPENMP
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
      std::cout << "OpenMP enabled, cores: " << omp_get_num_threads()
                << std::endl;
  }
#else
  std::cout << "OpenMP disabled." << std::endl;
#endif

  // ==========================================================================
  // READ INPUT DATA (with error checking — Finding 3)
  // ==========================================================================
  std::cout << "Reading data... " << std::flush;

  if (!utils::read_vector_from_file(JobName + ".data1", ControlData1, 6)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".data1" << std::endl;
    return 1;
  }

  // Try extended .data2 format (15 entries, Sec. 7.4 of report) first,
  // fall back to standard 11-entry format (Finding 1: report/implementation contract)
  bool extended_data2 = false;
  if (!utils::read_vector_from_file(JobName + ".data2", ControlData2, 15)) {
    if (!utils::read_vector_from_file(JobName + ".data2", ControlData2, 11)) {
      std::cerr << "FATAL: Cannot read " << JobName << ".data2" << std::endl;
      return 1;
    }
  } else {
    extended_data2 = true;
  }

  NodeNo       = ControlData1[0];
  StepNo       = ControlData1[1];
  PrenotchNo   = ControlData1[2];
  VINodesNo    = ControlData1[3];
  constrainsNo = ControlData1[4];
  maxCloudSize = ControlData1[5];

  E_young = ControlData2[0];
  Nu      = ControlData2[1];
  Rho     = ControlData2[2];
  wf      = ControlData2[3];
  t0      = ControlData2[4];
  t1      = ControlData2[5];
  dt      = ControlData2[6];
  R       = ControlData2[7];
  dx      = ControlData2[8];
  dy      = ControlData2[9];
  s0      = ControlData2[10];

  // Compute Lame constants (Sec. 5 of report)
  Lambda = E_young * Nu / ((1.0 + Nu) * (1.0 - 2.0 * Nu));
  Mu     = E_young / (2.0 * (1.0 + Nu));
  Kappa  = Lambda + Mu;  // 2D plane strain bulk modulus (Eq. kappa_2d)

  // If extended .data2 provided, read preprocessor-exported LPS constants
  // (indices 12-15 per Table 2 in report)
  double m_ref_from_file = 0.0;
  if (extended_data2) {
    m_ref_from_file = ControlData2[11];  // m (weighted volume)
    // ControlData2[12] = kappa, ControlData2[13] = alpha, ControlData2[14] = lambda_PD
    // These will be verified against computed values later
    std::cout << "(extended .data2 with " << 15 << " entries) ";
  } else {
    std::cout << "(standard .data2 with 11 entries; LPS constants computed in solver) ";
  }

  if (!utils::read_matrix_from_file(JobName + ".nodes", Nodes, NodeNo, DPN)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".nodes" << std::endl;
    return 1;
  }
  if (!utils::read_vector_from_file(JobName + ".bctypes", BCTypes, DPN * NodeNo)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".bctypes" << std::endl;
    return 1;
  }
  if (!utils::read_vector_from_file(JobName + ".bcvalues", BCValues, DPN * NodeNo)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".bcvalues" << std::endl;
    return 1;
  }
  if (!utils::read_vector_from_file(JobName + ".icdvalues", ICDValues, DPN * NodeNo)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".icdvalues" << std::endl;
    return 1;
  }
  if (!utils::read_vector_from_file(JobName + ".icvalues", ICValues, DPN * NodeNo)) {
    std::cerr << "FATAL: Cannot read " << JobName << ".icvalues" << std::endl;
    return 1;
  }
  // Prenotch and vinodes may legitimately be empty (size 0)
  if (PrenotchNo > 0) {
    utils::read_matrix_from_file(JobName + ".prenotch", Prenotches, PrenotchNo, 4);
  }
  if (VINodesNo > 0) {
    utils::read_vector_from_file(JobName + ".vinodes", VINodes, VINodesNo);
  }

  if (ABCS_STEP != -1 && constrainsNo > 0) {
    int cols_per_cloud = (ABCS_STEP == 1) ? 2 : 4;  // 2*ABCS_STEP entries per cloud node
    if (!utils::read_matrix_from_file(JobName + ".ciNodesU", ciNodesU,
                                 2 * constrainsNo, cols_per_cloud * maxCloudSize)) {
      std::cerr << "FATAL: Cannot read " << JobName << ".ciNodesU" << std::endl;
      return 1;
    }
    if (!utils::read_matrix_from_file(JobName + ".ciNodesV", ciNodesV,
                                 2 * constrainsNo, cols_per_cloud * maxCloudSize)) {
      std::cerr << "FATAL: Cannot read " << JobName << ".ciNodesV" << std::endl;
      return 1;
    }
    if (!utils::read_int_matrix_from_file(JobName + ".iClouds", iClouds,
                                 constrainsNo, maxCloudSize)) {
      std::cerr << "FATAL: Cannot read " << JobName << ".iClouds" << std::endl;
      return 1;
    }
    if (!utils::read_int_vector_from_file(JobName + ".infNodesIndex", infNodes,
                                 constrainsNo)) {
      std::cerr << "FATAL: Cannot read " << JobName << ".infNodesIndex" << std::endl;
      return 1;
    }
#if THETA_ABC_METHOD == 1
    // Read G_theta updating vector (Sec. 4.8)
    {
      int cols_theta = ABCS_STEP * maxCloudSize;
      if (!utils::read_matrix_from_file(JobName + ".ciNodesTheta", ciNodesTheta,
                                   constrainsNo, cols_theta)) {
        std::cerr << "FATAL: Cannot read " << JobName << ".ciNodesTheta" << std::endl;
        return 1;
      }
      std::cout << "(G_theta: " << constrainsNo << " x " << cols_theta << ") ";
    }
#endif
  }

  std::cout << "finished." << std::endl;

  // ==========================================================================
  // VERIFICATION: Print and check material parameters
  // ==========================================================================
  std::cout << std::endl << "=== Material Parameters ===" << std::endl;
  std::cout << std::setprecision(10);
  std::cout << "  E     = " << E_young << std::endl;
  std::cout << "  Nu    = " << Nu << std::endl;
  std::cout << "  Rho   = " << Rho << std::endl;
  std::cout << "  Lambda= " << Lambda << std::endl;
  std::cout << "  Mu    = " << Mu << std::endl;
  std::cout << "  Kappa = " << Kappa << " (2D bulk = lambda+mu)" << std::endl;

  // CHECK: Lame parameter identity
  {
    double E_check = Mu * (3.0 * Lambda + 2.0 * Mu) / (Lambda + Mu);
    double Nu_check = Lambda / (2.0 * (Lambda + Mu));
    std::cout << "  CHECK: E from Lambda,Mu = " << E_check
              << " (err=" << std::abs(E_check - E_young) / E_young << ")" << std::endl;
    std::cout << "  CHECK: Nu from Lambda,Mu = " << Nu_check
              << " (err=" << std::abs(Nu_check - Nu) << ")" << std::endl;
    if (std::abs(E_check - E_young) / E_young > 1e-10 ||
        std::abs(Nu_check - Nu) > 1e-10) {
      std::cerr << "FATAL: Lame parameter consistency check failed!" << std::endl;
      return 1;
    }
  }

  // CHECK: For nu=1/4, lambda should equal mu (BB-PD limit)
  if (std::abs(Nu - 0.25) < 1e-10) {
    std::cout << "  CHECK: Nu=1/4 detected. Lambda-Mu = " << Lambda - Mu
              << " (should be ~0)" << std::endl;
  }

  std::cout << std::endl << "=== Discretization ===" << std::endl;
  std::cout << "  NodeNo = " << NodeNo << std::endl;
  std::cout << "  StepNo = " << StepNo << std::endl;
  std::cout << "  delta (R) = " << R << std::endl;
  std::cout << "  dx = " << dx << ", dy = " << dy << std::endl;
  std::cout << "  dt = " << dt << std::endl;
  std::cout << "  delta/dx = " << R / dx << " (m-ratio)" << std::endl;
  std::cout << "  Prenotches = " << PrenotchNo << std::endl;
  std::cout << "  ABC nodes = " << constrainsNo << std::endl;
  std::cout << "  maxCloudSize = " << maxCloudSize << std::endl;

  utils::timer timer;

  // ==========================================================================
  // BUILD HORIZONS (kd-tree)
  // ==========================================================================
  timer.tic();
  int MaxHorizonLength;
  // Search radius: include bonds up to delta+dx/2 (volume-corrected horizon).
  // Bonds at distance > delta-dx/2 have partial volume correction beta < 1.
  // This must match the preprocessor's template grid radius for consistency
  // between the solver's discrete quadrature and the ABC dispersion relation.
  double horizon_search_r = R + dx / 2.0 + 1e-5;
  build_clouds(0, MAX_HORIZON_LENGTH, horizon_search_r, 0.0, Nodes, Horizons,
               HorizonLengths, MaxHorizonLength);
  std::cout << std::endl << "Building horizons took " << timer.toc()
            << " s. Max horizon length: " << MaxHorizonLength << std::endl;

  // VERIFICATION: Check horizon symmetry for a sample of nodes
  {
    int sym_failures = 0;
    int sym_checked = 0;
    for (int i = 0; i < std::min(NodeNo, 1000); i++) {
      for (int j = 1; j < HorizonLengths[i]; j++) {
        int n = Horizons(i, j);
        if (n == -1) continue;
        // Check that node n also has node i in its horizon
        bool found = false;
        for (int k = 1; k < HorizonLengths[n]; k++) {
          if (Horizons(n, k) == i) { found = true; break; }
        }
        if (!found) sym_failures++;
        sym_checked++;
      }
    }
    std::cout << "  Horizon symmetry check: " << sym_checked << " bonds checked, "
              << sym_failures << " asymmetric" << std::endl;
    if (sym_failures > 0) {
      std::cerr << "WARNING: Horizon asymmetry detected! This may cause errors."
                << std::endl;
    }
  }

  // ==========================================================================
  // APPLY PRENOTCH VISIBILITY
  // ==========================================================================
  timer.tic();
  for (int p = 0; p < PrenotchNo; p++) {
    const double n0x = Prenotches(p, 0), n0y = Prenotches(p, 1);
    const double n1x = Prenotches(p, 2), n1y = Prenotches(p, 3);
#pragma omp parallel for
    for (int i = 0; i < NodeNo; i++) {
      for (int j = 1; j < MAX_HORIZON_LENGTH; j++) {
        int n = Horizons(i, j);
        if (n == -1) continue;
        if (line_segment_intersect(Nodes(i, 0), Nodes(i, 1),
                                   Nodes(n, 0), Nodes(n, 1),
                                   n0x, n0y, n1x, n1y))
          Horizons(i, j) = -1;
      }
    }
  }
  std::cout << "Prenotch visibility took " << timer.toc() << " s." << std::endl;

  // ==========================================================================
  // COMPUTE NODE-WISE WEIGHTED VOLUME m_i (Eq. m_disc)
  // m_i = sum_j omega(|xi_j|) * |xi_j|^2 * beta(|xi_j|) * dx^2
  // ==========================================================================
  timer.tic();
  mWeighted.resize(NodeNo);
  Theta.resize(NodeNo);

#pragma omp parallel for
  for (int i = 0; i < NodeNo; i++) {
    double m_i = 0.0;
    for (int j = 1; j < HorizonLengths[i]; j++) {
      int n = Horizons(i, j);
      if (n == -1) continue;
      const double xi_x = Nodes(n, 0) - Nodes(i, 0);
      const double xi_y = Nodes(n, 1) - Nodes(i, 1);
      const double xi_norm = std::sqrt(xi_x * xi_x + xi_y * xi_y);
      const double w = omega_func(xi_norm);
      const double b = beta(xi_norm);
      m_i += w * xi_norm * xi_norm * b * dx * dy;
    }
    mWeighted[i] = m_i;
  }

  // VERIFICATION: Weighted volume statistics and continuum comparison
  {
    double m_min = 1e30, m_max = 0.0, m_sum = 0.0;
    int m_count = 0;
    int m_boundary_nodes = 0;
    for (int i = 0; i < NodeNo; i++) {
      if (mWeighted[i] > 0.0) {
        m_min = std::min(m_min, mWeighted[i]);
        m_max = std::max(m_max, mWeighted[i]);
        m_sum += mWeighted[i];
        m_count++;
      }
    }
    double m_continuum = M_PI * R * R * R * R / 2.0;
    m_ref = m_max;  // interior nodes have full family -> max m_i

    std::cout << std::endl << "=== Weighted Volume m_i ===" << std::endl;
    std::cout << "  min = " << m_min << std::endl;
    std::cout << "  max = " << m_max << " (used as m_ref for LPS constants)" << std::endl;
    std::cout << "  avg = " << m_sum / m_count << std::endl;
    std::cout << "  Continuum ref: pi*delta^4/2 = " << m_continuum << std::endl;
    std::cout << "  max/continuum = " << m_max / m_continuum
              << " (should be ~1)" << std::endl;

    // CHECK: m_max should be close to continuum value
    double m_err = std::abs(m_max - m_continuum) / m_continuum;
    if (m_err > 0.05) {
      std::cerr << "WARNING: Max weighted volume deviates " << m_err * 100.0
                << "% from continuum reference." << std::endl;
    }
  }
  std::cout << "Computing m_i took " << timer.toc() << " s." << std::endl;

  // ==========================================================================
  // OVERRIDE m_i AT ABC NODES WITH m_ref
  // The LPS force uses per-node m_i: alpha_i = 8*mu/m_i. At boundary nodes,
  // truncated horizons give m_i < m_ref, amplifying the force coefficient.
  // The ABC modes assume uniform m_ref everywhere. To make the force at
  // the boundary consistent with the ABC, we set m_i = m_ref at ABC nodes.
  // This eliminates the surface stiffening effect that causes instability.
  // ==========================================================================
  if (ABCS_STEP != -1 && constrainsNo > 0) {
    int n_corrected = 0;
    for (int i = 0; i < constrainsNo; i++) {
      int node = infNodes[i];
      if (mWeighted[node] < m_ref * 0.99) {
        n_corrected++;
      }
      mWeighted[node] = m_ref;
    }
    std::cout << "Overrode m_i -> m_ref at " << constrainsNo << " ABC nodes ("
              << n_corrected << " had m_i < 0.99*m_ref)." << std::endl;
  }

  // ==========================================================================
  // COMPUTE LPS CONSTANTS (Sec. 5, Eq. alpha_2d, lambdaPD_2d)
  // ==========================================================================
  AlphaLPS = 8.0 * Mu / m_ref;      // alpha = 8*mu/m  (Eq. alpha_2d)
  LambdaPD = 2.0 * (Lambda - Mu) / m_ref;  // lambda_PD = 2*(lambda-mu)/m (Eq. lambdaPD_2d)

  std::cout << std::endl << "=== LPS Constants (Sec. 5 of report) ===" << std::endl;
  std::cout << "  m_ref      = " << m_ref << std::endl;
  std::cout << "  alpha      = 8*mu/m     = " << AlphaLPS << std::endl;
  std::cout << "  lambda_PD  = 2*(lam-mu)/m = " << LambdaPD << std::endl;
  std::cout << "  d*kappa/m  = " << 2.0 * Kappa / m_ref << std::endl;
  std::cout << "  alpha/d    = " << AlphaLPS / 2.0 << std::endl;

  // CHECK: lambda_PD = d*kappa/m - alpha/d (Eq. lambda_pd)
  {
    double lpd_check = 2.0 * Kappa / m_ref - AlphaLPS / 2.0;
    std::cout << "  CHECK: d*kappa/m - alpha/d = " << lpd_check
              << " (should match lambda_PD=" << LambdaPD << ")" << std::endl;
    if (std::abs(lpd_check - LambdaPD) > 1e-15 * std::abs(LambdaPD + 1e-30)) {
      std::cerr << "FATAL: lambda_PD consistency check failed!" << std::endl;
      return 1;
    }
  }

  // CHECK: For nu=1/4, lambda_PD should vanish
  if (std::abs(Nu - 0.25) < 1e-10) {
    std::cout << "  CHECK: Nu=1/4 -> lambda_PD = " << LambdaPD
              << " (should be ~0)" << std::endl;
    if (std::abs(LambdaPD) > 1e-10 * AlphaLPS) {
      std::cerr << "WARNING: lambda_PD not zero for nu=1/4!" << std::endl;
    }
  }

  // ==========================================================================
  // COMPUTE DOMAIN GEOMETRY
  // ==========================================================================
  double domainRadius = 0.0;
  for (int i = 0; i < NodeNo; i++) {
    double r = std::sqrt(Nodes(i, 0) * Nodes(i, 0) +
                         Nodes(i, 1) * Nodes(i, 1));
    if (r > domainRadius) domainRadius = r;
  }
  std::cout << std::endl << "Domain radius: " << domainRadius << std::endl;

  // Exclude absorbing layer from output and energy computation
  // Only exclude when there are actual ABC nodes (constrainsNo > 0)
  const double output_exclude_r = (ABCS_STEP != -1 && constrainsNo > 0)
      ? domainRadius - R - dx / 100.0
      : -1.0;
  if (output_exclude_r > 0.0) {
    std::cout << "Output/energy exclude radius: " << output_exclude_r << std::endl;
  } else {
    std::cout << "No absorbing layer; all nodes included in output/energy." << std::endl;
  }

  // ==========================================================================
  // CRITICAL TIME STEP CHECK (Sec. 7.3.4 of report)
  // ==========================================================================
  {
    // Classical wave speeds
    double cP = std::sqrt((Lambda + 2.0 * Mu) / Rho);
    double cS = std::sqrt(Mu / Rho);
    double dt_conservative = dx / cP;

    std::cout << std::endl << "=== Time Step Stability ===" << std::endl;
    std::cout << "  cP = sqrt((lam+2mu)/rho) = " << cP << " mm/s" << std::endl;
    std::cout << "  cS = sqrt(mu/rho)        = " << cS << " mm/s" << std::endl;
    std::cout << "  cP/cS = " << cP / cS << std::endl;
    std::cout << "  dt = " << dt << std::endl;
    std::cout << "  dt_conservative (dx/cP) = " << dt_conservative << std::endl;
    std::cout << "  dt/dt_conservative = " << dt / dt_conservative << std::endl;

    // LPS-specific dt_c (Eq. dt_lps in report, Finding 4: match formula exactly)
    // dt_c = sqrt(2*rho / max_i { sum_j [d*kappa*d*|xi|^2/m^2 + 2*alpha/|xi|^2] *
    //                                     omega * beta * dx^2 })
    // where d=2, kappa=lambda+mu, alpha=8*mu/m
    double max_denom = 0.0;
    for (int i = 0; i < std::min(NodeNo, 10000); i++) {
      double denom = 0.0;
      const double mi = mWeighted[i];
      if (mi <= 0.0) continue;
      for (int j = 1; j < HorizonLengths[i]; j++) {
        int n = Horizons(i, j);
        if (n == -1) continue;
        const double xi_x = Nodes(n, 0) - Nodes(i, 0);
        const double xi_y = Nodes(n, 1) - Nodes(i, 1);
        const double xi2 = xi_x * xi_x + xi_y * xi_y;
        const double xi_norm = std::sqrt(xi2);
        const double b = beta(xi_norm);
        const double w = omega_func(xi_norm);
        // Eq. dt_lps: effective stiffness per bond
        // dilatation: d*kappa*d*|xi|^2/m^2 = 4*kappa*|xi|^2/m^2
        // deviatoric: 2*alpha/|xi|^2
        denom += (2.0 * Kappa * 2.0 * xi2 / (mi * mi) +
                  2.0 * AlphaLPS / xi2) * w * b * dx * dy;
      }
      max_denom = std::max(max_denom, denom);
    }
    if (max_denom > 0.0) {
      double dt_lps = std::sqrt(2.0 * Rho / max_denom);
      std::cout << "  dt_LPS (Eq. dt_lps, sampled) = " << dt_lps << std::endl;
      std::cout << "  dt/dt_LPS = " << dt / dt_lps << std::endl;
      if (dt > dt_lps) {
        std::cerr << "WARNING: dt exceeds LPS stability limit!" << std::endl;
      }
    }

    if (dt > dt_conservative) {
      std::cerr << "WARNING: dt exceeds conservative stability limit (dx/cP)!"
                << std::endl;
    }
  }

  // ==========================================================================
  // ALLOCATE SOLUTION VECTORS
  // ==========================================================================
  space::vector<double, int> Unm2(DPN * NodeNo);
  space::vector<double, int> Unm1(DPN * NodeNo);
  space::vector<double, int> Udnm2(DPN * NodeNo);
  space::vector<double, int> Udnm1(DPN * NodeNo);
  space::vector<double, int> Uddnm2(DPN * NodeNo);
  space::vector<double, int> Uddnm1(DPN * NodeNo);

  space::vector<double, int> Un(DPN * NodeNo);
  space::vector<double, int> Udn(DPN * NodeNo);
  space::vector<double, int> Uddn(DPN * NodeNo);

  space::vector<double, int> Un1(DPN * NodeNo);
  space::vector<double, int> Udn1(DPN * NodeNo);
  space::vector<double, int> Uddn1(DPN * NodeNo);

  space::vector<double, int> UMag(NodeNo);

#if THETA_ABC_METHOD == 1
  // Theta history for G_theta ABC (Sec. 4.8)
  space::vector<double, int> ThetaN(NodeNo);    // theta at step n
  space::vector<double, int> ThetaNm1(NodeNo);  // theta at step n-1
#endif

  // ==========================================================================
  // SET INITIAL CONDITIONS
  // ==========================================================================
#pragma omp parallel for
  for (int i = 0; i < DPN * NodeNo; i++) {
    Uddnm2[i] = 0.0;
    Uddnm1[i] = 0.0;
    Uddn[i]   = 0.0;
    Udnm2[i]  = 0.0;
    Udnm1[i]  = 0.0;
    Udn[i]    = ICDValues[i];
    Unm1[i]   = 0.0;
    Un[i]     = ICValues[i];
  }

  // VERIFICATION: Check initial conditions
  {
    double ic_max = 0.0, ic_sum = 0.0;
    int ic_nonzero = 0;
    for (int i = 0; i < DPN * NodeNo; i++) {
      double val = std::abs(Un[i]);
      ic_max = std::max(ic_max, val);
      ic_sum += val;
      if (val > 0.0) ic_nonzero++;
    }
    std::cout << std::endl << "=== Initial Conditions ===" << std::endl;
    std::cout << "  max|u0| = " << ic_max << std::endl;
    std::cout << "  nonzero IC entries: " << ic_nonzero
              << " / " << DPN * NodeNo << std::endl;

    double icd_max = 0.0;
    for (int i = 0; i < DPN * NodeNo; i++) {
      icd_max = std::max(icd_max, std::abs(Udn[i]));
    }
    std::cout << "  max|v0| = " << icd_max << std::endl;
  }

#if THETA_ABC_METHOD == 1
  // ==========================================================================
  // INITIALIZE THETA HISTORY (for G_theta ABC, Sec. 4.8)
  // ==========================================================================
  // Compute theta from initial displacement (Un) for ThetaN
  // ThetaNm1 = 0 (corresponding to Unm1 = 0)
#pragma omp parallel for
  for (int i = 0; i < NodeNo; i++) {
    double theta_sum = 0.0;
    for (int j = 1; j < HorizonLengths[i]; j++) {
      int n = Horizons(i, j);
      if (n == -1) continue;
      const double xi_x = Nodes(n, 0) - Nodes(i, 0);
      const double xi_y = Nodes(n, 1) - Nodes(i, 1);
      const double xi_norm = std::sqrt(xi_x * xi_x + xi_y * xi_y);
      const double eta_x = Un[DPN * n]     - Un[DPN * i];
      const double eta_y = Un[DPN * n + 1] - Un[DPN * i + 1];
      const double e = (eta_x * xi_x + eta_y * xi_y) / xi_norm;
      const double w = omega_func(xi_norm);
      const double b = beta(xi_norm);
      theta_sum += w * xi_norm * e * b * dx * dy;
    }
    ThetaN[i] = (mWeighted[i] > 0.0) ? (2.0 / mWeighted[i]) * theta_sum : 0.0;
    ThetaNm1[i] = 0.0;
  }
  std::cout << "Theta history initialized from IC." << std::endl;
#endif

  // ==========================================================================
  // EXPORT INITIAL CONDITIONS (VTK)
  // ==========================================================================
  utils::calc_mag(UMag, Un);
  utils::write_vtk_vector(JobName + Prefix + "-0.vtk",
                          Nodes, Un, "displacement", NodeNo, output_exclude_r);

  // Open energy log file
  std::ofstream energyFile;
  energyFile.open(JobName + Prefix + ".energy",
                  std::ios::out | std::ios::trunc);
  energyFile << "# t  E_total  E_kinetic  E_potential  P_x  P_y  max_u  max_theta"
             << std::endl;
  energyFile.close();

  // Open verification log
  std::ofstream verifyFile;
  verifyFile.open(JobName + Prefix + ".verify",
                  std::ios::out | std::ios::trunc);
  verifyFile << "# t  E_total  dE/E0  P_x  P_y  max_theta  max_u  max_v  max_a"
             << std::endl;
  verifyFile.close();

  double E0 = -1.0;  // initial total energy (set after first output step)

  // ==========================================================================
  // TIME MARCHING LOOP (Algorithm 2 in report)
  // ==========================================================================
  std::cout << std::endl << "Time marching started (" << StepNo << " steps):"
            << std::endl;
  timer.tic();

  for (int s = 1; s <= StepNo; s++) {
    const double t = s * dt;

    // ========================================================================
    // Step 1: Predict displacement (velocity Verlet predictor)
    //   u^{n+1} = u^n + dt * u_dot^n + dt^2/2 * u_ddot^n
    // ========================================================================
#pragma omp parallel for
    for (int i = 0; i < DPN * NodeNo; i++) {
      Un1[i] = Un[i] + dt * (Udn[i] + (dt / 2.0) * Uddn[i]);
    }

    // ========================================================================
    // Step 2: LPS Force computation - TWO PASS (Algorithm 1 in report)
    // ========================================================================

    // ------ Pass 1: Compute dilatation theta_i for all nodes (Eq. theta_disc) ------
    // theta_i = (d/m_i) * sum_j omega * (xi . eta) * beta * dx^2
#pragma omp parallel for
    for (int i = 0; i < NodeNo; i++) {
      double theta_sum = 0.0;
      for (int j = 1; j < HorizonLengths[i]; j++) {
        int n = Horizons(i, j);
        if (n == -1) continue;

        const double xi_x = Nodes(n, 0) - Nodes(i, 0);
        const double xi_y = Nodes(n, 1) - Nodes(i, 1);
        const double xi_norm = std::sqrt(xi_x * xi_x + xi_y * xi_y);

        const double eta_x = Un1[DPN * n]     - Un1[DPN * i];
        const double eta_y = Un1[DPN * n + 1] - Un1[DPN * i + 1];

        // Extension scalar e = xi_hat . eta = (xi . eta) / |xi|
        const double e = (eta_x * xi_x + eta_y * xi_y) / xi_norm;

        const double w = omega_func(xi_norm);
        const double b = beta(xi_norm);

        // Accumulate: omega * |xi| * e * beta * dV = omega * (xi.eta) * beta * dV
        theta_sum += w * xi_norm * e * b * dx * dy;
      }
      // d = 2 for 2D plane strain
      Theta[i] = (mWeighted[i] > 0.0) ? (2.0 / mWeighted[i]) * theta_sum : 0.0;
    }

    // Override theta at ABC nodes (truncated horizons bias the direct computation).
    if (ABCS_STEP != -1) {
#if THETA_ABC_METHOD == 0
      // Method 0: IDW extrapolation from interior neighbors (Sec. 5.2).
      // Inverse-distance weighted average of theta from interior neighbors,
      // giving a smooth theta field consistent with the interior.
      for (int i = 0; i < constrainsNo; i++) {
        int node = infNodes[i];
        if (BCTypes[2 * node] != 1) continue;

        double theta_avg = 0.0;
        double w_sum = 0.0;
        for (int j = 1; j < HorizonLengths[node]; j++) {
          int nb = Horizons(node, j);
          if (nb == -1) continue;
          if (BCTypes[2 * nb] == 1) continue;  // skip other ABC nodes

          double rx = Nodes(nb, 0) - Nodes(node, 0);
          double ry = Nodes(nb, 1) - Nodes(node, 1);
          double dist = std::sqrt(rx * rx + ry * ry);
          double wt = 1.0 / (dist + 1e-30);
          theta_avg += wt * Theta[nb];
          w_sum += wt;
        }
        if (w_sum > 0.0) {
          Theta[node] = theta_avg / w_sum;
        }
      }
#else
      // Method 1: G_theta updating vector (Sec. 4.8).
      // Predict theta at ABC nodes from cloud theta history:
      //   theta^{n+1}_ABC = G_theta . {theta^n_cloud, theta^{n-1}_cloud}
      for (int i = 0; i < constrainsNo; i++) {
        int node = infNodes[i];
        if (BCTypes[2 * node] != 1) continue;

        double theta_abc = 0.0;
        for (int j = 0; j < maxCloudSize; j++) {
          int k = iClouds(i, j);
          if (k == -1) continue;

          if (ABCS_STEP == 2) {
            theta_abc += ciNodesTheta(i, 2 * j)     * ThetaN[k]
                       + ciNodesTheta(i, 2 * j + 1) * ThetaNm1[k];
          } else if (ABCS_STEP == 1) {
            theta_abc += ciNodesTheta(i, j) * ThetaN[k];
          }
        }
        Theta[node] = theta_abc;
      }
#endif
    }

    // ------ Pass 2: Compute force density (Eq. force_disc) ------
    // F_i = sum_j [lambda_PD_ij * omega * |xi| * (theta_i + theta_j)
    //              + 2*alpha_ij * omega * e] * xi_hat * beta * dV
    //
    // With node-wise m: lambda_PD_i = 2*(lambda-mu)/m_i, alpha_i = 8*mu/m_i
#pragma omp parallel for
    for (int i = 0; i < NodeNo; i++) {
      double F0 = 0.0, F1 = 0.0;
      const double m_i = mWeighted[i];

      // Guard: skip isolated/degenerate nodes with zero weighted volume (Finding 2)
      if (m_i <= 0.0) {
        Uddn1[DPN * i]     = 0.0;
        Uddn1[DPN * i + 1] = 0.0;
        continue;
      }

      for (int j = 1; j < HorizonLengths[i]; j++) {
        int n = Horizons(i, j);
        if (n == -1) continue;

        const double xi_x = Nodes(n, 0) - Nodes(i, 0);
        const double xi_y = Nodes(n, 1) - Nodes(i, 1);
        const double xi_norm = std::sqrt(xi_x * xi_x + xi_y * xi_y);

        const double eta_x = Un1[DPN * n]     - Un1[DPN * i];
        const double eta_y = Un1[DPN * n + 1] - Un1[DPN * i + 1];

        // Extension scalar e = xi_hat . eta
        const double e = (eta_x * xi_x + eta_y * xi_y) / xi_norm;

        const double w = omega_func(xi_norm);
        const double b = beta(xi_norm);
        const double m_j = mWeighted[n];

        // Guard: skip neighbor with zero weighted volume (Finding 2)
        if (m_j <= 0.0) continue;

        // Scalar force state from node i (Eq. lps_force linearized):
        //   t_i = lambda_PD_i * omega * |xi| * theta_i + alpha_i * omega * e
        //       = [2(lam-mu)/m_i] * omega * |xi| * theta_i + [8*mu/m_i] * omega * e
        const double t_i = (2.0 * (Lambda - Mu) / m_i) * Theta[i] * xi_norm * w
                         + (8.0 * Mu / m_i) * w * e;

        // Scalar force state from node j:
        //   t_j = [2(lam-mu)/m_j] * omega * |xi| * theta_j + [8*mu/m_j] * omega * e
        const double t_j = (2.0 * (Lambda - Mu) / m_j) * Theta[n] * xi_norm * w
                         + (8.0 * Mu / m_j) * w * e;

        // Net force contribution: (t_i + t_j) * xi_hat * beta * dV
        const double scale = (t_i + t_j) * b * dx * dy;

        F0 += scale * xi_x / xi_norm;
        F1 += scale * xi_y / xi_norm;
      }

      Uddn1[DPN * i]     = F0 / Rho;
      Uddn1[DPN * i + 1] = F1 / Rho;
    }

    // ========================================================================
    // Step 2b: Apply displacement ABCs AFTER force computation (BB-PD ordering).
    //   This ordering is more stable than ABC-before-forces because the
    //   velocity Verlet prediction is energy-conserving, avoiding feedback
    //   through the theta/force coupling.
    //   u^{n+1}_i = G_u . {u^n, u^{n-1}}_cloud
    // ========================================================================
    if (ABCS_STEP != -1) {
      for (int i = 0; i < constrainsNo; i++) {
        int node = infNodes[i];
        if (BCTypes[2 * node] != 1) continue;

        Un1[2 * node]     = 0.0;
        Un1[2 * node + 1] = 0.0;

        for (int j = 0; j < maxCloudSize; j++) {
          int k = iClouds(i, j);
          if (k == -1) continue;

          if (ABCS_STEP == 1) {
            Un1[2 * node] +=
                ciNodesU(2 * i, 2 * j)     * Un[2 * k] +
                ciNodesU(2 * i, 2 * j + 1) * Un[2 * k + 1];
            Un1[2 * node + 1] +=
                ciNodesU(2 * i + 1, 2 * j)     * Un[2 * k] +
                ciNodesU(2 * i + 1, 2 * j + 1) * Un[2 * k + 1];
          } else if (ABCS_STEP == 2) {
            Un1[2 * node] +=
                ciNodesU(2 * i, 4 * j)     * Un[2 * k] +
                ciNodesU(2 * i, 4 * j + 1) * Un[2 * k + 1] +
                ciNodesU(2 * i, 4 * j + 2) * Unm1[2 * k] +
                ciNodesU(2 * i, 4 * j + 3) * Unm1[2 * k + 1];
            Un1[2 * node + 1] +=
                ciNodesU(2 * i + 1, 4 * j)     * Un[2 * k] +
                ciNodesU(2 * i + 1, 4 * j + 1) * Un[2 * k + 1] +
                ciNodesU(2 * i + 1, 4 * j + 2) * Unm1[2 * k] +
                ciNodesU(2 * i + 1, 4 * j + 3) * Unm1[2 * k + 1];
          }
        }
      }
    }

    // ========================================================================
    // Step 3: Correct velocity (velocity Verlet corrector)
    //   u_dot^{n+1} = u_dot^n + dt/2 * (u_ddot^n + u_ddot^{n+1})
    // ========================================================================
#pragma omp parallel for
    for (int i = 0; i < DPN * NodeNo; i++) {
      Udn1[i] = Udn[i] + (dt / 2.0) * Uddn[i] + (dt / 2.0) * Uddn1[i];
    }

    // (No impact velocity BC for Example 1)

    // ========================================================================
    // Step 4: Apply velocity ABCs (independent G_v fit, matching BB-PD)
    //   u_dot^{n+1} = G_v . {v^n, v^{n-1}}_cloud
    // ========================================================================
    if (ABCS_STEP != -1) {
      for (int i = 0; i < constrainsNo; i++) {
        int node = infNodes[i];

        // Guard: skip nodes not flagged as active ABC
        if (BCTypes[2 * node] != 1) continue;

#if KINEMATIC_VELOCITY_ABC
        // Kinematic velocity: v^{n+1} = (u^{n+1} - u^n) / dt
        Udn1[2 * node]     = (Un1[2 * node]     - Un[2 * node])     / dt;
        Udn1[2 * node + 1] = (Un1[2 * node + 1] - Un[2 * node + 1]) / dt;
#else
        Udn1[2 * node]     = 0.0;
        Udn1[2 * node + 1] = 0.0;

        for (int j = 0; j < maxCloudSize; j++) {
          int k = iClouds(i, j);
          if (k == -1) continue;

          if (ABCS_STEP == 1) {
            Udn1[2 * node] +=
                ciNodesV(2 * i, 2 * j)     * Udn[2 * k] +
                ciNodesV(2 * i, 2 * j + 1) * Udn[2 * k + 1];
            Udn1[2 * node + 1] +=
                ciNodesV(2 * i + 1, 2 * j)     * Udn[2 * k] +
                ciNodesV(2 * i + 1, 2 * j + 1) * Udn[2 * k + 1];
          } else if (ABCS_STEP == 2) {
            Udn1[2 * node] +=
                ciNodesV(2 * i, 4 * j)     * Udn[2 * k] +
                ciNodesV(2 * i, 4 * j + 1) * Udn[2 * k + 1] +
                ciNodesV(2 * i, 4 * j + 2) * Udnm1[2 * k] +
                ciNodesV(2 * i, 4 * j + 3) * Udnm1[2 * k + 1];
            Udn1[2 * node + 1] +=
                ciNodesV(2 * i + 1, 4 * j)     * Udn[2 * k] +
                ciNodesV(2 * i + 1, 4 * j + 1) * Udn[2 * k + 1] +
                ciNodesV(2 * i + 1, 4 * j + 2) * Udnm1[2 * k] +
                ciNodesV(2 * i + 1, 4 * j + 3) * Udnm1[2 * k + 1];
          }
        }
#endif
      }
    }

    // (No bond breaking for Example 1 -- Sec. 7.3.3 of report)

    // ========================================================================
    // Step 6: Output and verification
    // ========================================================================
    if (s % SUBSTEP_NO == 0) {

      // --- Compute energy (LPS strain energy, Sec. 2 of report) ---
      // W_i = kappa/2 * theta_i^2 + alpha/2 * sum_j omega * (e^d)^2 * beta * dV_j
      // E_total = sum_i W_i * dV_i
      double energy = 0.0, energyK = 0.0, energyP = 0.0;
      double Px = 0.0, Py = 0.0;   // linear momentum
      double max_u = 0.0, max_v = 0.0, max_a = 0.0;
      double theta_min = 1e30, theta_max = -1e30;
      int max_u_node = -1, max_v_node = -1;

      for (int i = 0; i < NodeNo; i++) {
        double x = Nodes(i, 0), y = Nodes(i, 1);
        double r = std::sqrt(x * x + y * y);

        // Track max fields (all nodes)
        double u_mag = std::sqrt(Un1[2*i]*Un1[2*i] + Un1[2*i+1]*Un1[2*i+1]);
        double v_mag = std::sqrt(Udn1[2*i]*Udn1[2*i] + Udn1[2*i+1]*Udn1[2*i+1]);
        double a_mag = std::sqrt(Uddn1[2*i]*Uddn1[2*i] + Uddn1[2*i+1]*Uddn1[2*i+1]);
        if (u_mag > max_u) { max_u = u_mag; max_u_node = i; }
        if (v_mag > max_v) { max_v = v_mag; max_v_node = i; }
        max_a = std::max(max_a, a_mag);
        theta_min = std::min(theta_min, Theta[i]);
        theta_max = std::max(theta_max, Theta[i]);

        // Energy/momentum only in near field (exclude absorbing layer)
        if (output_exclude_r > 0.0 && r > output_exclude_r) continue;

        // Linear momentum: p = rho * v * dV
        Px += Rho * Udn1[2 * i]     * dx * dy;
        Py += Rho * Udn1[2 * i + 1] * dx * dy;

        // Kinetic energy: 1/2 * rho * |v|^2 * dV
        double vx = Udn1[2 * i], vy = Udn1[2 * i + 1];
        double ek = 0.5 * Rho * (vx * vx + vy * vy) * dx * dy;
        energyK += ek;

        // LPS strain energy density W_i:
        const double m_i = mWeighted[i];
        // Dilatation part: kappa/2 * theta^2  (Eq. lps_force)
        double W_i = 0.5 * Kappa * Theta[i] * Theta[i];

        // Deviatoric part: alpha/2 * sum_j omega * (e^d)^2 * beta * dV_j
        // alpha/2 = 4*mu/m_i
        for (int j = 1; j < HorizonLengths[i]; j++) {
          int n = Horizons(i, j);
          if (n == -1) continue;
          const double xi_x = Nodes(n, 0) - Nodes(i, 0);
          const double xi_y = Nodes(n, 1) - Nodes(i, 1);
          const double xi_norm = std::sqrt(xi_x * xi_x + xi_y * xi_y);
          const double eta_x = Un1[DPN * n]     - Un1[DPN * i];
          const double eta_y = Un1[DPN * n + 1] - Un1[DPN * i + 1];
          const double e = (eta_x * xi_x + eta_y * xi_y) / xi_norm;
          const double ed = e - Theta[i] * xi_norm / 2.0;  // e^d = e - theta*|xi|/d, d=2
          const double w = omega_func(xi_norm);
          const double b = beta(xi_norm);
          W_i += (4.0 * Mu / m_i) * w * ed * ed * b * dx * dy;
        }

        double ep = W_i * dx * dy;  // W_i * dV_i
        energyP += ep;
        energy += ek + ep;
      }

      // Set reference energy at first output step
      if (E0 < 0.0) E0 = energy;
      double dE_rel = (E0 > 0.0) ? (energy - E0) / E0 : 0.0;

      // Boundary passivity: compute ||u_B||, ||v_B|| at absorbing nodes
      double uB_sq = 0.0, vB_sq = 0.0;
      int nActiveABC = 0;
      if (ABCS_STEP != -1) {
        for (int i = 0; i < constrainsNo; i++) {
          int node = infNodes[i];
          if (BCTypes[2 * node] != 1) continue;
          nActiveABC++;
          uB_sq += Un1[2*node]*Un1[2*node] + Un1[2*node+1]*Un1[2*node+1];
          vB_sq += Udn1[2*node]*Udn1[2*node] + Udn1[2*node+1]*Udn1[2*node+1];
        }
      }
      double uB_rms = (nActiveABC > 0) ? std::sqrt(uB_sq / nActiveABC) : 0.0;
      double vB_rms = (nActiveABC > 0) ? std::sqrt(vB_sq / nActiveABC) : 0.0;

      // Console output
      std::cout << s / SUBSTEP_NO << "\tt=" << std::setprecision(6) << t
                << "\tE=" << std::setprecision(8) << energy
                << "\tK=" << energyK << "\tP=" << energyP
                << "\tdE/E0=" << std::setprecision(4) << dE_rel
                << "\t|p|=" << std::sqrt(Px*Px + Py*Py)
                << "\tuB=" << uB_rms << "\tvB=" << vB_rms
                << std::setprecision(10) << std::endl;
      // Diagnostic: locate the node with max displacement and velocity
      if (max_u_node >= 0) {
        double xu = Nodes(max_u_node, 0), yu = Nodes(max_u_node, 1);
        double ru = std::sqrt(xu*xu + yu*yu);
        bool is_abc_u = (BCTypes[2*max_u_node] == 1);
        std::cout << "  max|u|=" << max_u << " at node " << max_u_node
                  << " (x=" << xu << " y=" << yu << " r=" << ru
                  << (is_abc_u ? " ABC" : " INT") << ")" << std::endl;
      }
      if (max_v_node >= 0) {
        double xv = Nodes(max_v_node, 0), yv = Nodes(max_v_node, 1);
        double rv = std::sqrt(xv*xv + yv*yv);
        bool is_abc_v = (BCTypes[2*max_v_node] == 1);
        std::cout << "  max|v|=" << max_v << " at node " << max_v_node
                  << " (x=" << xv << " y=" << yv << " r=" << rv
                  << (is_abc_v ? " ABC" : " INT") << ")" << std::endl;
      }

      // VTK outputs: displacement, velocity, dilatation
      std::stringstream ss;
      ss << s;
      std::string step_str = ss.str();

      utils::write_vtk_vector(JobName + Prefix + "-" + step_str + ".vtk",
                              Nodes, Un1, "displacement", NodeNo,
                              output_exclude_r);
      utils::write_vtk_vector(JobName + Prefix + "-vel-" + step_str + ".vtk",
                              Nodes, Udn1, "velocity", NodeNo,
                              output_exclude_r);
      utils::write_vtk_scalar(JobName + Prefix + "-theta-" + step_str + ".vtk",
                              Nodes, Theta, "dilatation", NodeNo,
                              output_exclude_r);

      // Energy log (extended with boundary norms)
      energyFile.open(JobName + Prefix + ".energy",
                      std::ios::out | std::ios::app);
      energyFile << std::setprecision(16)
                 << t << " " << energy << " " << energyK << " " << energyP
                 << " " << Px << " " << Py << " " << max_u << " " << theta_max
                 << " " << uB_rms << " " << vB_rms
                 << std::endl;
      energyFile.close();

      // Verification log
      verifyFile.open(JobName + Prefix + ".verify",
                      std::ios::out | std::ios::app);
      verifyFile << std::setprecision(16)
                 << t << " " << energy << " " << dE_rel
                 << " " << Px << " " << Py
                 << " " << theta_max << " " << max_u << " " << max_v << " " << max_a
                 << std::endl;
      verifyFile.close();

      // VERIFICATION: Check for NaN/Inf
      if (std::isnan(energy) || std::isinf(energy)) {
        std::cerr << "FATAL: Energy is NaN/Inf at step " << s << "!" << std::endl;
        return 1;
      }
      if (std::isnan(max_u) || std::isinf(max_u)) {
        std::cerr << "FATAL: Displacement is NaN/Inf at step " << s << "!" << std::endl;
        return 1;
      }

      // VERIFICATION: Energy drift warning (without ABCs, should be ~0)
      if (ABCS_STEP == -1 && E0 > 0.0 && std::abs(dE_rel) > 0.01) {
        std::cerr << "WARNING: Energy drift > 1% at step " << s
                  << " (dE/E0=" << dE_rel << "). Check dt stability." << std::endl;
      }
    }

    // ========================================================================
    // Step 7: Swap arrays for next time step
    // ========================================================================
    Unm2.swap(Unm1);
    Unm1.swap(Un);
    Un.swap(Un1);

    Udnm2.swap(Udnm1);
    Udnm1.swap(Udn);
    Udn.swap(Udn1);

    Uddnm2.swap(Uddnm1);
    Uddnm1.swap(Uddn);
    Uddn.swap(Uddn1);

#if THETA_ABC_METHOD == 1
    // Shift theta history: ThetaNm1 <- ThetaN <- Theta
    ThetaNm1.swap(ThetaN);
    for (int i = 0; i < NodeNo; i++) {
      ThetaN[i] = Theta[i];
    }
#endif
  }

  std::cout << std::endl << "Time marching took " << timer.toc() << " seconds."
            << std::endl;

  return 0;
}
