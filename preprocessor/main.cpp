// =============================================================================
// LPS-PD ABC Preprocessor
// =============================================================================
// Generates all input data files for the PD2_LPS_ABC solver, including
// the absorbing boundary condition (ABC) updating vectors G_u and G_v.
//
// This replaces the Mathematica notebook used in the BB-PD paper.
// Key steps:
//   1. Generate circular domain (Example 1)
//   2. Identify absorbing boundary nodes
//   3. Build interpolation clouds for each absorbing node
//   4. Compute LPS dispersion matrix M^LPS for each wave vector mode
//   5. Solve 2x2 eigenvalue problem -> omega_P, omega_S, gamma_P, gamma_S
//   6. Construct plane wave basis functions
//   7. Solve least-squares for G_u, G_v per absorbing node
//   8. Export all data files
//
// References:
//   - LPS_ABC_Report.tex (Sec. 4-7)
//   - Shojaei et al., "PD elastic waves in 2D unbounded domains" (CMAME)

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

// LAPACK least-squares solvers
extern "C" {
  // SVD-based least-squares (handles rank-deficient systems)
  void dgelsd_(int *m, int *n, int *nrhs,
               double *a, int *lda, double *b, int *ldb,
               double *s, double *rcond, int *rank,
               double *work, int *lwork, int *iwork, int *info);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Configuration
// =============================================================================

// Material: Steel 18Ni1900 (same as BB-PD paper)
static const double MAT_E   = 190.0 / 1000.0;     // Young's modulus [kN/mm^2]
static const double MAT_NU  = 0.25;                // Poisson's ratio
static const double MAT_RHO = 8000.0 / 1e18;       // density [kg/mm^3]
static const double MAT_G0  = 22170.0 * 1e-9;      // fracture energy [J/mm^2]

// Discretization
static const double DOMAIN_RADIUS = 125.0;         // circular domain radius [mm]
static const double DX = 0.5;                      // grid spacing x [mm]
static const double DY = 0.5;                      // grid spacing y [mm]
static const double DELTA = 4.0 * DX;              // horizon radius [mm] (m-ratio=4)
static const double DT = 17.5e-9;                  // time step [s] — halved to test dt hypothesis
static const int    N_STEPS = 12000;                 // total time steps (doubled for halved dt)

// ABC parameters (from paper: Sec. 4 & Examples)
static const int    ABCS_STEP = 2;                  // temporal stencil (2-step)
// Paper values: Delta_phi = 0.0392 rad, Delta_kappa = 0.00043 rad/mm
// Stability criterion: ||G_u|| <= 1 and ||G_v|| <= 1
// Matching Mathematica: theta in Dphi*[-1,1] step Dphi/5, kappa in Dk*[0,1] step Dk/5
static const double MODE_DELTA_PHI   = 0.0392;      // angular range [rad]
static const int    MODE_N_SUB       = 5;            // subdivisions per direction
static const double MODE_DELTA_KAPPA_INIT = 0.00043; // initial wavenumber range [rad/mm]

// SVD truncation threshold: dgelsd rcond parameter
// Singular values below rcond * max_singular_value are treated as zero.
// rcond = -1: machine epsilon (matches Mathematica LeastSquares default)
// With cloud-adaptive mode count ensuring underdetermined systems,
// minimum-norm solution keeps ||G|| bounded without aggressive truncation.
static const double SVD_RCOND = 1e-14;                  // "wow" run value — good absorption, slight amplification

// Global mode count factor:
// nModesGlobal = min(total_modes, floor(MODE_ETA * nDataRows_typical / 4))
// With MODE_ETA >= 1.15, all modes are used (matching BB-PD paper).
// With MODE_ETA < 1.0, modes are reduced to make system underdetermined.
static const double MODE_ETA = 2.0;                   // use all modes (BB-PD-like)

// =============================================================================
// Derived material constants
// =============================================================================
struct MaterialConstants {
  double E, nu, rho, G0;
  double lambda, mu, kappa;
  double s0;        // critical stretch
  double c_P, c_S;  // wave speeds
  double m_ref;     // reference weighted volume
  double alpha_lps; // 8*mu/m
  double lambda_pd; // 2*(lambda-mu)/m
};

MaterialConstants compute_material() {
  MaterialConstants mc;
  mc.E   = MAT_E;
  mc.nu  = MAT_NU;
  mc.rho = MAT_RHO;
  mc.G0  = MAT_G0;

  mc.lambda = mc.E * mc.nu / ((1.0 + mc.nu) * (1.0 - 2.0 * mc.nu));
  mc.mu     = mc.E / (2.0 * (1.0 + mc.nu));
  mc.kappa  = mc.lambda + mc.mu;

  mc.s0 = std::sqrt(5.0 * M_PI * mc.G0 / (12.0 * mc.E * DELTA));

  mc.c_P = std::sqrt((mc.lambda + 2.0 * mc.mu) / mc.rho);
  mc.c_S = std::sqrt(mc.mu / mc.rho);

  // m_ref, alpha_lps, lambda_pd computed after TempGrid is built
  mc.m_ref = 0.0;
  mc.alpha_lps = 0.0;
  mc.lambda_pd = 0.0;

  return mc;
}

// =============================================================================
// Volume correction factor beta (Eq. beta in report)
// =============================================================================
inline double beta_func(double xi, double delta, double dx) {
  if (xi <= delta - dx / 2.0)
    return 1.0;
  else if (xi <= delta + dx / 2.0)
    return (delta + dx / 2.0 - xi) / dx;
  else
    return 0.0;
}

// =============================================================================
// Build the template grid (canonical neighborhood for an interior node)
// =============================================================================
struct Vec2 { double x, y; };

std::vector<Vec2> build_template_grid(double delta, double dx, double dy) {
  std::vector<Vec2> grid;
  double rmax = delta + dx / 2.0;
  // Loop must be lattice-aligned (multiples of dx) to match the actual grid.
  // Extend one ring beyond delta to capture bonds up to delta+dx/2.
  double loop_max = delta + dx;
  for (double xi_x = -loop_max; xi_x <= loop_max + 1e-10; xi_x += dx) {
    for (double xi_y = -loop_max; xi_y <= loop_max + 1e-10; xi_y += dy) {
      double r = std::sqrt(xi_x * xi_x + xi_y * xi_y);
      // Include bonds up to delta+dx/2 (same cutoff as the solver).
      // Bonds at r>delta-dx/2 have partial volume correction beta<1.
      // This must match the solver's horizon search radius for consistency.
      if (r > 1e-12 && r <= rmax + 1e-10) {
        Vec2 v;
        v.x = xi_x;
        v.y = xi_y;
        grid.push_back(v);
      }
    }
  }
  return grid;
}

// =============================================================================
// Compute reference weighted volume m_ref from TempGrid
// =============================================================================
double compute_m_ref(const std::vector<Vec2> &tg, double delta, double dx, double dy) {
  double m = 0.0;
  for (size_t j = 0; j < tg.size(); j++) {
    double r = std::sqrt(tg[j].x * tg[j].x + tg[j].y * tg[j].y);
    double b = beta_func(r, delta, dx);
    m += r * r * b * dx * dy;  // omega=1
  }
  return m;
}

// =============================================================================
// Wave mode data
// =============================================================================
struct WaveMode {
  double alpha1, alpha2;   // wave vector
  double omega_P, omega_S; // P-wave and S-wave frequencies
  double gamma_P[2];       // P-wave polarization
  double gamma_S[2];       // S-wave polarization
  double S_real[2], S_imag[2]; // complex S vector for G_theta (Sec. 4.8)
};

// =============================================================================
// Compute LPS dispersion matrix and solve eigenvalue problem
// for a single wave vector alpha = (a1, a2)
// =============================================================================
WaveMode solve_dispersion(double a1, double a2,
                          const MaterialConstants &mc,
                          const std::vector<Vec2> &tg) {
  WaveMode mode;
  mode.alpha1 = a1;
  mode.alpha2 = a2;

  // Compute S_k and N_jk integrals (Eq. Sk_disc, Npq_disc)
  std::complex<double> S[2] = {{0,0}, {0,0}};
  std::complex<double> N[2][2] = {{{0,0},{0,0}},{{0,0},{0,0}}};

  for (size_t j = 0; j < tg.size(); j++) {
    double xi_x = tg[j].x;
    double xi_y = tg[j].y;
    double xi2 = xi_x * xi_x + xi_y * xi_y;
    double xi_norm = std::sqrt(xi2);
    double b = beta_func(xi_norm, DELTA, DX);
    double dV = DX * DY;

    double phase = a1 * xi_x + a2 * xi_y;
    std::complex<double> exp_factor(std::cos(phase) - 1.0, std::sin(phase));

    // S_k (Eq. Sk_disc)
    S[0] += xi_x * exp_factor * b * dV;
    S[1] += xi_y * exp_factor * b * dV;

    // N_jk (Eq. Npq_disc)
    double xi_comp[2] = {xi_x, xi_y};
    for (int p = 0; p < 2; p++) {
      for (int k = 0; k < 2; k++) {
        N[p][k] += (xi_comp[p] * xi_comp[k] / xi2) * exp_factor * b * dV;
      }
    }
  }

  // Store S vector in mode for G_theta computation (Sec. 4.8)
  mode.S_real[0] = S[0].real(); mode.S_imag[0] = S[0].imag();
  mode.S_real[1] = S[1].real(); mode.S_imag[1] = S[1].imag();

  // Build M^LPS (Eq. M_LPS): M_pk = (d*lambda_pd/m)*S_p*S_k + 2*alpha*N_pk
  // d=2, lambda_pd = 2*(lambda-mu)/m, alpha = 8*mu/m
  double coeff_S = 2.0 * mc.lambda_pd / mc.m_ref;  // d*lambda_pd/m
  double coeff_N = 2.0 * mc.alpha_lps;              // 2*alpha

  double M[2][2];
  for (int p = 0; p < 2; p++) {
    for (int k = 0; k < 2; k++) {
      std::complex<double> val = coeff_S * S[p] * S[k] + coeff_N * N[p][k];
      M[p][k] = val.real();
      // Verify imaginary part is negligible (relative to trace or absolute 1e-15)
      double imag_tol = std::max(1e-8 * (std::abs(M[0][0]) + std::abs(M[1][1])), 1e-15);
      if (std::abs(val.imag()) > imag_tol) {
        std::cerr << "WARNING: M[" << p << "][" << k << "] has imaginary part "
                  << val.imag() << " (tol=" << imag_tol << ")" << std::endl;
      }
    }
  }

  // Solve 2x2 eigenvalue problem: det(M + rho*omega^2*I) = 0
  // Eigenvalues of M: lambda = (tr +/- sqrt(tr^2 - 4*det)) / 2
  double tr = M[0][0] + M[1][1];
  double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  double disc = tr * tr - 4.0 * det;
  if (disc < 0.0) disc = 0.0;  // numerical noise

  double eig1 = (tr + std::sqrt(disc)) / 2.0;  // closer to 0 (less negative)
  double eig2 = (tr - std::sqrt(disc)) / 2.0;  // more negative

  // omega^2 = -eigenvalue / rho
  double omP2 = -eig2 / mc.rho;  // P-wave (larger omega)
  double omS2 = -eig1 / mc.rho;  // S-wave (smaller omega)

  mode.omega_P = (omP2 > 0.0) ? std::sqrt(omP2) : 0.0;
  mode.omega_S = (omS2 > 0.0) ? std::sqrt(omS2) : 0.0;

  // Eigenvectors via null space of (M - eig*I)
  // For 2x2: if A = [[a,b],[c,d]], nullspace of A ~ [-b, a-eig] or [d-eig, -c]
  auto compute_eigvec = [&](double eig, double *gamma) {
    double a00 = M[0][0] - eig;
    double a01 = M[0][1];
    double a10 = M[1][0];
    double a11 = M[1][1] - eig;
    // Use the row with larger magnitude for stability
    double r0 = std::abs(a00) + std::abs(a01);
    double r1 = std::abs(a10) + std::abs(a11);
    if (r0 >= r1) {
      gamma[0] = -a01;
      gamma[1] = a00;
    } else {
      gamma[0] = a11;
      gamma[1] = -a10;
    }
    double norm = std::sqrt(gamma[0] * gamma[0] + gamma[1] * gamma[1]);
    if (norm > 1e-30) {
      gamma[0] /= norm;
      gamma[1] /= norm;
    }
  };

  // Handle degenerate case: alpha=0 gives M=0, both eigenvalues zero.
  // In this case, any two orthogonal unit vectors are valid eigenvectors.
  // Mathematica's NullSpace[ZeroMatrix[2]] returns {{1,0},{0,1}}.
  double M_norm = std::abs(M[0][0]) + std::abs(M[0][1]) +
                  std::abs(M[1][0]) + std::abs(M[1][1]);
  if (M_norm < 1e-30) {
    mode.gamma_P[0] = 1.0; mode.gamma_P[1] = 0.0;
    mode.gamma_S[0] = 0.0; mode.gamma_S[1] = 1.0;
  } else {
    compute_eigvec(eig2, mode.gamma_P);
    compute_eigvec(eig1, mode.gamma_S);
  }

  return mode;
}

// =============================================================================
// Build mode grid and compute all dispersion relations
// Matches Mathematica: theta in Dphi*[-1,1] step Dphi/n_sub,
//                      kappa in [0, delta_kappa] step delta_kappa/n_sub
// =============================================================================
std::vector<WaveMode> compute_all_modes(const MaterialConstants &mc,
                                         const std::vector<Vec2> &tg,
                                         double delta_kappa) {
  std::vector<WaveMode> modes;

  // Matching Mathematica convention (Sec. 8.2 of report):
  //   {IModeLo,IModeHi,IModeStep} = DeltaTheta * {-1, 1, 1/n_sub}
  //   {IkLo,  IkHi,  IkStep}     = DeltaKappa * { 0, 1, 1/n_sub}
  double d_phi = MODE_DELTA_PHI;
  int n_sub = MODE_N_SUB;

  for (int i_theta = -n_sub; i_theta <= n_sub; i_theta++) {
    double theta = d_phi * static_cast<double>(i_theta) / n_sub;
    for (int i_k = 0; i_k <= n_sub; i_k++) {
      double kappa_val = delta_kappa * static_cast<double>(i_k) / n_sub;

      double a1 = kappa_val * std::cos(theta);
      double a2 = kappa_val * std::sin(theta);

      // Check for duplicates (same alpha vector within tolerance)
      bool duplicate = false;
      for (size_t m = 0; m < modes.size(); m++) {
        double da1 = modes[m].alpha1 - a1;
        double da2 = modes[m].alpha2 - a2;
        if (std::sqrt(da1 * da1 + da2 * da2) < 1e-12) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) continue;

      WaveMode wm = solve_dispersion(a1, a2, mc, tg);
      modes.push_back(wm);
    }
  }

  return modes;
}

// =============================================================================
// LAPACK dgels wrapper: solve A*x = B in least-squares sense
// A is m x n (column-major), B is m x nrhs (overwritten with solution)
// =============================================================================
struct LSResult {
  int info;
  int rank;
  double s_max, s_min;  // largest and smallest singular values
};

LSResult solve_least_squares(int m, int n, int nrhs,
                        std::vector<double> &A, // m*n, column-major
                        std::vector<double> &B, // max(m,n)*nrhs, column-major
                        int ldb,
                        double rcond_in = -99.0) {
  int lda = m;
  int info = 0;
  int rank = 0;
  double rcond = (rcond_in > -90.0) ? rcond_in : SVD_RCOND;
  int min_mn = std::min(m, n);
  std::vector<double> s(min_mn);

  // Query optimal workspace
  int lwork = -1;
  double wkopt;
  int iwork_query;
  dgelsd_(&m, &n, &nrhs, A.data(), &lda, B.data(), &ldb,
          s.data(), &rcond, &rank, &wkopt, &lwork, &iwork_query, &info);
  lwork = static_cast<int>(wkopt);
  std::vector<double> work(lwork);
  int nlvl = std::max(1, static_cast<int>(std::log2(min_mn / 26.0)) + 1);
  int liwork = std::max(1, 3 * min_mn * nlvl + 11 * min_mn);
  std::vector<int> iwork(liwork);

  // Solve using SVD
  dgelsd_(&m, &n, &nrhs, A.data(), &lda, B.data(), &ldb,
          s.data(), &rcond, &rank, work.data(), &lwork, iwork.data(), &info);

  LSResult res;
  res.info = info;
  res.rank = rank;
  res.s_max = (min_mn > 0) ? s[0] : 0.0;
  res.s_min = (min_mn > 0) ? s[min_mn - 1] : 0.0;
  return res;
}

// =============================================================================
// Compute ABC updating vectors for one absorbing node
// =============================================================================
// psi_displacement: gamma * cos/sin(alpha.x_local - omega*t)
// psi_velocity:     gamma * omega * sin/cos(alpha.x_local - omega*t)
//                   (d/dt of displacement basis)
//
// Returns G_u and G_v as flat arrays of size 2 * (2*ABCS_STEP*nCloud)
// Layout per row: [Un_x(j0), Un_y(j0), Unm1_x(j0), Unm1_y(j0),
//                  Un_x(j1), Un_y(j1), Unm1_x(j1), Unm1_y(j1), ...]
// =============================================================================
void compute_abc_node(
    int abc_idx,
    double abc_x, double abc_y,        // absorbing node position
    double tau,                         // outward normal angle
    const std::vector<int> &cloud,      // cloud node indices
    const std::vector<double> &all_nodes_x, // all node x-coords
    const std::vector<double> &all_nodes_y, // all node y-coords
    const std::vector<WaveMode> &modes,
    const MaterialConstants &mc,
    int maxCloudSize,
    std::vector<double> &G_u_row0,  // output: x-component row
    std::vector<double> &G_u_row1,  // output: y-component row
    std::vector<double> &G_v_row0,
    std::vector<double> &G_v_row1,
    double rcond_override = -99.0,  // optional: override SVD_RCOND
    int nModesUse = -1,             // optional: limit number of modes used
    double dt_use = DT)             // optional: override time step
{
  int nCloud = static_cast<int>(cloud.size());
  int nModes_total = static_cast<int>(modes.size());
  int nModes = (nModesUse > 0 && nModesUse < nModes_total) ? nModesUse : nModes_total;
  int nBasis = 4 * nModes;  // cos_P, cos_S, sin_P, sin_S per mode

  // Rotation: local coords where x-axis = outward normal
  double cosT = std::cos(tau);
  double sinT = std::sin(tau);

  // Data matrix rows: 2*nCloud per time level, ABCS_STEP time levels
  int nDataRows = 2 * nCloud * ABCS_STEP;

  // We solve: A^T * x = target  (A^T is nBasis x nDataRows)
  // This means A is nDataRows x nBasis
  // Using dgels with A^T (m=nBasis, n=nDataRows): overdetermined if nBasis > nDataRows

  int m_ls = nBasis;     // rows of A^T
  int n_ls = nDataRows;  // cols of A^T
  int ldb = std::max(m_ls, n_ls);

  // Build A^T (column-major, m_ls x n_ls)
  // A^T[basis_idx][data_idx] = psi evaluated at cloud data point
  std::vector<double> AT_u(m_ls * n_ls, 0.0);  // for displacement
  std::vector<double> AT_v(m_ls * n_ls, 0.0);  // for velocity

  // Helper: column-major index for m_ls x n_ls matrix
  // A^T[row][col] stored at row + col*m_ls
  auto at_idx = [&](int row, int col) -> int { return row + col * m_ls; };

  for (int im = 0; im < nModes; im++) {
    double a1 = modes[im].alpha1;
    double a2 = modes[im].alpha2;
    double omP = modes[im].omega_P;
    double omS = modes[im].omega_S;
    const double *gP = modes[im].gamma_P;
    const double *gS = modes[im].gamma_S;

    // Time levels: t_level=0 -> t=0, t_level=1 -> t=-dt
    for (int t_level = 0; t_level < ABCS_STEP; t_level++) {
      double t_val = -t_level * dt_use;  // t=0 or t=-dt

      for (int jc = 0; jc < nCloud; jc++) {
        // Cloud node position relative to absorbing node
        double dx_g = all_nodes_x[cloud[jc]] - abc_x;
        double dy_g = all_nodes_y[cloud[jc]] - abc_y;

        // Rotate to local frame
        double x_loc =  cosT * dx_g + sinT * dy_g;
        double y_loc = -sinT * dx_g + cosT * dy_g;

        double phase = a1 * x_loc + a2 * y_loc;

        // P-wave phases
        double phase_P = phase - omP * t_val;
        double phase_S = phase - omS * t_val;

        // Displacement basis in local coords, then rotate back to global
        // Basis 0: gamma_P * cos(phase_P)
        // Basis 1: gamma_S * cos(phase_S)
        // Basis 2: gamma_P * sin(phase_P)
        // Basis 3: gamma_S * sin(phase_S)
        double cos_P = std::cos(phase_P);
        double sin_P = std::sin(phase_P);
        double cos_S = std::cos(phase_S);
        double sin_S = std::sin(phase_S);

        // Data row index for this cloud node and time level
        int row_base = t_level * (2 * nCloud) + 2 * jc;

        // Displacement basis functions (4 per mode)
        struct { double local_x, local_y; } disp_basis[4], vel_basis[4];

        // Basis 0: gamma_P * cos(phase_P)
        disp_basis[0].local_x = gP[0] * cos_P;
        disp_basis[0].local_y = gP[1] * cos_P;
        // Basis 1: gamma_S * cos(phase_S)
        disp_basis[1].local_x = gS[0] * cos_S;
        disp_basis[1].local_y = gS[1] * cos_S;
        // Basis 2: gamma_P * sin(phase_P)
        disp_basis[2].local_x = gP[0] * sin_P;
        disp_basis[2].local_y = gP[1] * sin_P;
        // Basis 3: gamma_S * sin(phase_S)
        disp_basis[3].local_x = gS[0] * sin_S;
        disp_basis[3].local_y = gS[1] * sin_S;

        // Velocity basis = d/dt of displacement basis
        // d/dt [gamma * cos(phase - omega*t)] = gamma * omega * sin(phase - omega*t)
        // d/dt [gamma * sin(phase - omega*t)] = -gamma * omega * cos(phase - omega*t)
        vel_basis[0].local_x = gP[0] * omP * sin_P;
        vel_basis[0].local_y = gP[1] * omP * sin_P;
        vel_basis[1].local_x = gS[0] * omS * sin_S;
        vel_basis[1].local_y = gS[1] * omS * sin_S;
        vel_basis[2].local_x = -gP[0] * omP * cos_P;
        vel_basis[2].local_y = -gP[1] * omP * cos_P;
        vel_basis[3].local_x = -gS[0] * omS * cos_S;
        vel_basis[3].local_y = -gS[1] * omS * cos_S;

        for (int b = 0; b < 4; b++) {
          int basis_row = 4 * im + b;

          // Rotate local -> global: [cosT, -sinT; sinT, cosT] * local
          double ux = cosT * disp_basis[b].local_x - sinT * disp_basis[b].local_y;
          double uy = sinT * disp_basis[b].local_x + cosT * disp_basis[b].local_y;

          // A^T[basis_row, row_base]   = u_x component
          // A^T[basis_row, row_base+1] = u_y component
          AT_u[at_idx(basis_row, row_base)]     = ux;
          AT_u[at_idx(basis_row, row_base + 1)] = uy;

          double vx = cosT * vel_basis[b].local_x - sinT * vel_basis[b].local_y;
          double vy = sinT * vel_basis[b].local_x + cosT * vel_basis[b].local_y;

          AT_v[at_idx(basis_row, row_base)]     = vx;
          AT_v[at_idx(basis_row, row_base + 1)] = vy;
        }
      }
    }
  }

  // Build target vectors: modes evaluated at origin (x_local=0,0) at t=+dt
  // target is nBasis x 2 (x and y components), stored column-major with ldb rows
  std::vector<double> target_u(ldb * 2, 0.0);
  std::vector<double> target_v(ldb * 2, 0.0);

  for (int im = 0; im < nModes; im++) {
    double omP = modes[im].omega_P;
    double omS = modes[im].omega_S;
    const double *gP = modes[im].gamma_P;
    const double *gS = modes[im].gamma_S;

    // At origin: phase = 0, t = +dt
    // Basis 0: gamma_P * cos(-omega_P * dt)
    double cos_Pdt = std::cos(-omP * dt_use);
    double sin_Pdt = std::sin(-omP * dt_use);
    double cos_Sdt = std::cos(-omS * dt_use);
    double sin_Sdt = std::sin(-omS * dt_use);

    struct { double lx, ly; } tgt_disp[4], tgt_vel[4];

    tgt_disp[0] = {gP[0] * cos_Pdt, gP[1] * cos_Pdt};
    tgt_disp[1] = {gS[0] * cos_Sdt, gS[1] * cos_Sdt};
    tgt_disp[2] = {gP[0] * sin_Pdt, gP[1] * sin_Pdt};
    tgt_disp[3] = {gS[0] * sin_Sdt, gS[1] * sin_Sdt};

    // velocity at origin at t=dt:
    // d/dt[gamma*cos(-omega*t)] at t=dt = gamma*omega*sin(-omega*dt)
    // d/dt[gamma*sin(-omega*t)] at t=dt = -gamma*omega*cos(-omega*dt)
    tgt_vel[0] = {gP[0] * omP * sin_Pdt, gP[1] * omP * sin_Pdt};
    tgt_vel[1] = {gS[0] * omS * sin_Sdt, gS[1] * omS * sin_Sdt};
    tgt_vel[2] = {-gP[0] * omP * cos_Pdt, -gP[1] * omP * cos_Pdt};
    tgt_vel[3] = {-gS[0] * omS * cos_Sdt, -gS[1] * omS * cos_Sdt};

    for (int b = 0; b < 4; b++) {
      int basis_row = 4 * im + b;
      // Rotate local -> global
      double ux = cosT * tgt_disp[b].lx - sinT * tgt_disp[b].ly;
      double uy = sinT * tgt_disp[b].lx + cosT * tgt_disp[b].ly;
      target_u[basis_row]       = ux;  // col 0 (x-component)
      target_u[basis_row + ldb] = uy;  // col 1 (y-component)

      double vx = cosT * tgt_vel[b].lx - sinT * tgt_vel[b].ly;
      double vy = sinT * tgt_vel[b].lx + cosT * tgt_vel[b].ly;
      target_v[basis_row]       = vx;
      target_v[basis_row + ldb] = vy;
    }
  }

  // Solve least-squares with SVD truncation (rcond controls which modes are kept)
  // A^T is m_ls x n_ls, target is m_ls x 2, G is n_ls x 2
  LSResult res_u = solve_least_squares(m_ls, n_ls, 2, AT_u, target_u, ldb, rcond_override);
  LSResult res_v = solve_least_squares(m_ls, n_ls, 2, AT_v, target_v, ldb, rcond_override);

  if (res_u.info != 0) {
    std::cerr << "WARNING: dgelsd for G_u failed at ABC node " << abc_idx
              << " (info=" << res_u.info << ")" << std::endl;
  }
  if (res_v.info != 0) {
    std::cerr << "WARNING: dgelsd for G_v failed at ABC node " << abc_idx
              << " (info=" << res_v.info << ")" << std::endl;
  }

  // Extract G from overwritten target (first n_ls rows)
  int cols = 2 * ABCS_STEP * maxCloudSize;
  G_u_row0.assign(cols, 0.0);
  G_u_row1.assign(cols, 0.0);
  G_v_row0.assign(cols, 0.0);
  G_v_row1.assign(cols, 0.0);

  for (int jc = 0; jc < nCloud; jc++) {
    for (int tl = 0; tl < ABCS_STEP; tl++) {
      int data_base = tl * (2 * nCloud) + 2 * jc;
      int out_col = 2 * ABCS_STEP * jc + 2 * tl;

      G_u_row0[out_col]     = target_u[data_base];
      G_u_row0[out_col + 1] = target_u[data_base + 1];
      G_u_row1[out_col]     = target_u[data_base + ldb];
      G_u_row1[out_col + 1] = target_u[data_base + 1 + ldb];

      G_v_row0[out_col]     = target_v[data_base];
      G_v_row0[out_col + 1] = target_v[data_base + 1];
      G_v_row1[out_col]     = target_v[data_base + ldb];
      G_v_row1[out_col + 1] = target_v[data_base + 1 + ldb];
    }
  }
}

// =============================================================================
// Compute ABC updating vector for THETA (dilatation) at one absorbing node
// (Sec. 4.8 of report)
// =============================================================================
// For a plane wave u = gamma * exp(i*(alpha.x - omega*t)), the linearized LPS
// dilatation (Seleson & Parks 2011) is:
//   theta = (2/m_ref) * (gamma . S) * exp(i*(alpha.x - omega*t))
// where S_k = sum_j xi_k * (exp(i*alpha.xi) - 1) * beta * dV  (from Sec. 3.2).
//
// theta_hat = (2/m_ref) * (gamma . S)  [complex scalar amplitude]
//
// Theta basis functions (4 per mode, scalar):
//   Re[theta_hat_P * exp(i*phase_P)], Re[theta_hat_S * exp(i*phase_S)],
//   Im[theta_hat_P * exp(i*phase_P)], Im[theta_hat_S * exp(i*phase_S)]
//
// Returns G_theta as flat array of size ABCS_STEP * maxCloudSize
// Layout: [ThetaN(j0), ThetaNm1(j0), ThetaN(j1), ThetaNm1(j1), ...]
// =============================================================================
void compute_abc_node_theta(
    int abc_idx,
    double abc_x, double abc_y,
    double tau,
    const std::vector<int> &cloud,
    const std::vector<double> &all_nodes_x,
    const std::vector<double> &all_nodes_y,
    const std::vector<WaveMode> &modes,
    const MaterialConstants &mc,
    int maxCloudSize,
    std::vector<double> &G_theta_out,
    double rcond_override = -99.0,
    int nModesUse = -1,
    double dt_use = DT)
{
  int nCloud = static_cast<int>(cloud.size());
  int nModes_total = static_cast<int>(modes.size());
  int nModes = (nModesUse > 0 && nModesUse < nModes_total) ? nModesUse : nModes_total;
  int nBasis = 4 * nModes;

  double cosT = std::cos(tau);
  double sinT = std::sin(tau);

  // Scalar theta: nCloud entries per time level (not 2*nCloud as for vectors)
  int nDataRows = nCloud * ABCS_STEP;

  int m_ls = nBasis;
  int n_ls = nDataRows;
  int ldb = std::max(m_ls, n_ls);

  // Build A^T (column-major, m_ls x n_ls)
  std::vector<double> AT_theta(m_ls * n_ls, 0.0);
  auto at_idx = [&](int row, int col) -> int { return row + col * m_ls; };

  for (int im = 0; im < nModes; im++) {
    double a1 = modes[im].alpha1;
    double a2 = modes[im].alpha2;
    double omP = modes[im].omega_P;
    double omS = modes[im].omega_S;
    const double *gP = modes[im].gamma_P;
    const double *gS = modes[im].gamma_S;

    // Complex theta amplitudes: theta_hat = (2/m_ref) * (gamma . S)
    double coeff = 2.0 / mc.m_ref;
    double thP_r = coeff * (gP[0] * modes[im].S_real[0] + gP[1] * modes[im].S_real[1]);
    double thP_i = coeff * (gP[0] * modes[im].S_imag[0] + gP[1] * modes[im].S_imag[1]);
    double thS_r = coeff * (gS[0] * modes[im].S_real[0] + gS[1] * modes[im].S_real[1]);
    double thS_i = coeff * (gS[0] * modes[im].S_imag[0] + gS[1] * modes[im].S_imag[1]);

    for (int t_level = 0; t_level < ABCS_STEP; t_level++) {
      double t_val = -t_level * dt_use;

      for (int jc = 0; jc < nCloud; jc++) {
        double dx_g = all_nodes_x[cloud[jc]] - abc_x;
        double dy_g = all_nodes_y[cloud[jc]] - abc_y;
        double x_loc =  cosT * dx_g + sinT * dy_g;
        double y_loc = -sinT * dx_g + cosT * dy_g;

        double phase = a1 * x_loc + a2 * y_loc;
        double phase_P = phase - omP * t_val;
        double phase_S = phase - omS * t_val;
        double cos_P = std::cos(phase_P);
        double sin_P = std::sin(phase_P);
        double cos_S = std::cos(phase_S);
        double sin_S = std::sin(phase_S);

        int data_row = t_level * nCloud + jc;

        // Theta basis: Re[theta_hat * exp(i*phase)] and Im[...]
        double theta_b[4];
        theta_b[0] = thP_r * cos_P - thP_i * sin_P;
        theta_b[1] = thS_r * cos_S - thS_i * sin_S;
        theta_b[2] = thP_r * sin_P + thP_i * cos_P;
        theta_b[3] = thS_r * sin_S + thS_i * cos_S;

        for (int b = 0; b < 4; b++) {
          AT_theta[at_idx(4 * im + b, data_row)] = theta_b[b];
        }
      }
    }
  }

  // Build target: theta at origin (x_local=0) at t=+dt
  std::vector<double> target_theta(ldb, 0.0);

  for (int im = 0; im < nModes; im++) {
    double omP = modes[im].omega_P;
    double omS = modes[im].omega_S;
    const double *gP = modes[im].gamma_P;
    const double *gS = modes[im].gamma_S;

    double coeff = 2.0 / mc.m_ref;
    double thP_r = coeff * (gP[0] * modes[im].S_real[0] + gP[1] * modes[im].S_real[1]);
    double thP_i = coeff * (gP[0] * modes[im].S_imag[0] + gP[1] * modes[im].S_imag[1]);
    double thS_r = coeff * (gS[0] * modes[im].S_real[0] + gS[1] * modes[im].S_real[1]);
    double thS_i = coeff * (gS[0] * modes[im].S_imag[0] + gS[1] * modes[im].S_imag[1]);

    double cos_Pdt = std::cos(-omP * dt_use);
    double sin_Pdt = std::sin(-omP * dt_use);
    double cos_Sdt = std::cos(-omS * dt_use);
    double sin_Sdt = std::sin(-omS * dt_use);

    target_theta[4 * im + 0] = thP_r * cos_Pdt - thP_i * sin_Pdt;
    target_theta[4 * im + 1] = thS_r * cos_Sdt - thS_i * sin_Sdt;
    target_theta[4 * im + 2] = thP_r * sin_Pdt + thP_i * cos_Pdt;
    target_theta[4 * im + 3] = thS_r * sin_Sdt + thS_i * cos_Sdt;
  }

  // Solve least-squares: A^T * G_theta = target_theta
  int nrhs = 1;
  LSResult res = solve_least_squares(m_ls, n_ls, nrhs, AT_theta, target_theta, ldb, rcond_override);

  if (res.info != 0) {
    std::cerr << "WARNING: dgelsd for G_theta failed at ABC node " << abc_idx
              << " (info=" << res.info << ")" << std::endl;
  }

  // Extract G_theta (first n_ls entries of overwritten target)
  int cols = ABCS_STEP * maxCloudSize;
  G_theta_out.assign(cols, 0.0);

  for (int jc = 0; jc < nCloud; jc++) {
    for (int tl = 0; tl < ABCS_STEP; tl++) {
      int data_idx = tl * nCloud + jc;
      int out_col = ABCS_STEP * jc + tl;
      G_theta_out[out_col] = target_theta[data_idx];
    }
  }
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char *argv[]) {
  std::cout << "============================================" << std::endl;
  std::cout << " LPS-PD ABC Preprocessor (Example 1)" << std::endl;
  std::cout << "============================================" << std::endl;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <jobname> [nu]" << std::endl;
    return 0;
  }
  std::string jobname = argv[1];
  double nu_override = -1.0;
  if (argc >= 3) {
    nu_override = std::atof(argv[2]);
    std::cout << "  Poisson ratio override: nu = " << nu_override << std::endl;
  }

  // ========================================================================
  // 1. Material constants
  // ========================================================================
  MaterialConstants mc = compute_material();
  if (nu_override > 0.0 && nu_override < 0.5) {
    mc.nu = nu_override;
    mc.lambda = mc.E * mc.nu / ((1.0 + mc.nu) * (1.0 - 2.0 * mc.nu));
    mc.mu     = mc.E / (2.0 * (1.0 + mc.nu));
    mc.kappa  = mc.lambda + mc.mu;
    mc.c_P = std::sqrt((mc.lambda + 2.0 * mc.mu) / mc.rho);
    mc.c_S = std::sqrt(mc.mu / mc.rho);
  }

  std::cout << std::endl << "=== Material ===" << std::endl;
  std::cout << "  E     = " << mc.E << std::endl;
  std::cout << "  nu    = " << mc.nu << std::endl;
  std::cout << "  rho   = " << mc.rho << std::endl;
  std::cout << "  lambda= " << mc.lambda << std::endl;
  std::cout << "  mu    = " << mc.mu << std::endl;
  std::cout << "  kappa = " << mc.kappa << std::endl;
  std::cout << "  c_P   = " << mc.c_P << " mm/s" << std::endl;
  std::cout << "  c_S   = " << mc.c_S << " mm/s" << std::endl;

  // ========================================================================
  // 2. Template grid and weighted volume
  // ========================================================================
  std::vector<Vec2> tempGrid = build_template_grid(DELTA, DX, DY);
  mc.m_ref = compute_m_ref(tempGrid, DELTA, DX, DY);
  mc.alpha_lps = 8.0 * mc.mu / mc.m_ref;
  mc.lambda_pd = 2.0 * (mc.lambda - mc.mu) / mc.m_ref;

  std::cout << std::endl << "=== LPS Constants ===" << std::endl;
  std::cout << "  Template grid: " << tempGrid.size() << " bonds" << std::endl;
  std::cout << "  m_ref     = " << mc.m_ref << std::endl;
  std::cout << "  m_cont    = " << M_PI * DELTA * DELTA * DELTA * DELTA / 2.0 << std::endl;
  std::cout << "  alpha_lps = " << mc.alpha_lps << std::endl;
  std::cout << "  lambda_pd = " << mc.lambda_pd << std::endl;

  // ========================================================================
  // 3. Generate circular domain
  // ========================================================================
  std::cout << std::endl << "Generating domain (R=" << DOMAIN_RADIUS
            << ", dx=" << DX << ")..." << std::flush;

  std::vector<double> nodes_x, nodes_y;
  for (double x = -DOMAIN_RADIUS; x <= DOMAIN_RADIUS + 1e-10; x += DX) {
    for (double y = -DOMAIN_RADIUS; y <= DOMAIN_RADIUS + 1e-10; y += DY) {
      if (std::sqrt(x * x + y * y) <= DOMAIN_RADIUS + 1e-10) {
        nodes_x.push_back(x);
        nodes_y.push_back(y);
      }
    }
  }
  int NodeNo = static_cast<int>(nodes_x.size());
  std::cout << " " << NodeNo << " nodes." << std::endl;

  // ========================================================================
  // 4. Identify absorbing boundary nodes (r > R - delta)
  // ========================================================================
  std::vector<int> abc_indices;    // global node indices
  std::vector<double> abc_tau;     // outward normal angle
  for (int i = 0; i < NodeNo; i++) {
    double r = std::sqrt(nodes_x[i] * nodes_x[i] + nodes_y[i] * nodes_y[i]);
    if (r > DOMAIN_RADIUS - DELTA) {
      abc_indices.push_back(i);
      abc_tau.push_back(std::atan2(nodes_y[i], nodes_x[i]));
    }
  }
  int constrainsNo = static_cast<int>(abc_indices.size());
  std::cout << "ABC nodes: " << constrainsNo << std::endl;

  // ========================================================================
  // 5. Build clouds for each absorbing node (neighbors within delta)
  // ========================================================================
  // Cloud radius = delta: full cloud has 49 nodes, nDataRows=196.
  // With per-node mode adaptation (nModes_i = nCloud_i), the LS system is
  // exactly square (nBasis = nDataRows) at each node → unique solution with
  // ||G|| ≈ 1 regardless of cloud size.
  double cloud_radius = DELTA;
  std::cout << "Building ABC clouds (radius=" << cloud_radius << ")..." << std::flush;
  std::vector<std::vector<int>> clouds(constrainsNo);
  int maxCloudSize = 0;

  for (int i = 0; i < constrainsNo; i++) {
    double ax = nodes_x[abc_indices[i]];
    double ay = nodes_y[abc_indices[i]];
    for (int j = 0; j < NodeNo; j++) {
      double dx_n = nodes_x[j] - ax;
      double dy_n = nodes_y[j] - ay;
      double dist = std::sqrt(dx_n * dx_n + dy_n * dy_n);
      if (dist <= cloud_radius + 1e-10) {
        clouds[i].push_back(j);
      }
    }
    maxCloudSize = std::max(maxCloudSize, static_cast<int>(clouds[i].size()));
  }
  std::cout << " done. maxCloudSize = " << maxCloudSize << std::endl;

  // ========================================================================
  // 6. Compute initial delta_kappa from LPS P-wave speed (Eq. 53 in report)
  //    Matching BB-PD: delta_kappa = 1/sqrt(c_P)
  // ========================================================================
  double delta_kappa = 1.0 / std::sqrt(mc.c_P);  // Eq. 53: c_P = sqrt((lam+2mu)/rho)

  std::cout << std::endl << "=== ABC Mode Parameters ===" << std::endl;
  std::cout << "  Delta_phi   = " << MODE_DELTA_PHI << " rad" << std::endl;
  std::cout << "  n_sub       = " << MODE_N_SUB << std::endl;
  std::cout << "  c_P^LPS     = " << mc.c_P << " mm/s" << std::endl;
  std::cout << "  Delta_kappa_init = " << delta_kappa << " rad/mm" << std::endl;

  // ========================================================================
  // 6a. Compute mode set at paper delta_kappa.
  //     With MODE_N_SUB=4 and cloud size ~49, the system should be
  //     overdetermined (nBasis < nDataRows), producing stable G norms.
  // ========================================================================
  std::vector<WaveMode> modes = compute_all_modes(mc, tempGrid, delta_kappa);
  std::cout << std::endl << "=== Mode Set ===" << std::endl;
  std::cout << "  dk = " << delta_kappa << std::endl;
  std::cout << "  nModes = " << modes.size() << std::endl;
  std::cout << "  nBasis = " << 4 * modes.size() << std::endl;
  int nDataRows_typ = 2 * maxCloudSize * ABCS_STEP;  // typical overdetermination
  std::cout << "  nDataRows (typical) = " << nDataRows_typ << std::endl;
  std::cout << "  overdetermination ratio = "
            << (double)nDataRows_typ / (4.0 * modes.size()) << std::endl;

  // Print sample modes
  {
    std::cout << std::endl << "=== Sample Modes (dk=" << delta_kappa << ") ===" << std::endl;
    for (int i = 0; i < std::min(static_cast<int>(modes.size()), 5); i++) {
      std::cout << "  mode " << i << ": alpha=(" << modes[i].alpha1 << ", "
                << modes[i].alpha2 << ") omega_P=" << modes[i].omega_P
                << " omega_S=" << modes[i].omega_S
                << " gamma_P=(" << modes[i].gamma_P[0] << ", "
                << modes[i].gamma_P[1] << ")"
                << " gamma_S=(" << modes[i].gamma_S[0] << ", "
                << modes[i].gamma_S[1] << ")" << std::endl;
    }
    if (modes.size() > 1) {
      double k0 = std::sqrt(modes[1].alpha1 * modes[1].alpha1 +
                             modes[1].alpha2 * modes[1].alpha2);
      if (k0 > 0) {
        std::cout << "  c_P_eff(mode1) = omega_P/k = "
                  << modes[1].omega_P / k0 << " (classical c_P = " << mc.c_P << ")"
                  << std::endl;
      }
    }
  }

  // ========================================================================
  // 7. Compute G_u and G_v for all absorbing nodes
  //    Cloud-adaptive mode count: for each node, limit nBasis so that
  //    nBasis <= eta * nDataRows, ensuring an underdetermined LS system.
  //    DGELSD then returns the minimum-norm solution, minimizing ||G||.
  // ========================================================================
  static const double G_NORM_TARGET = 1.0;

  // Sort modes by |kappa| ascending so we can trim high-kappa modes first.
  // Low-kappa modes capture long-wavelength waves; high-kappa modes are
  // more oscillatory and contribute less to the far-field representation.
  std::sort(modes.begin(), modes.end(), [](const WaveMode &a, const WaveMode &b) {
    double ka = a.alpha1 * a.alpha1 + a.alpha2 * a.alpha2;
    double kb = b.alpha1 * b.alpha1 + b.alpha2 * b.alpha2;
    return ka < kb;
  });

  std::cout << std::endl << "Modes sorted by |kappa| ascending." << std::endl;
  std::cout << "  mode  0: |kappa|=" << std::sqrt(modes[0].alpha1*modes[0].alpha1 + modes[0].alpha2*modes[0].alpha2) << std::endl;
  std::cout << "  mode " << modes.size()-1 << ": |kappa|=" << std::sqrt(modes.back().alpha1*modes.back().alpha1 + modes.back().alpha2*modes.back().alpha2) << std::endl;

  int cols_per_node = 2 * ABCS_STEP * maxCloudSize;
  std::vector<double> ciNodesU(2 * constrainsNo * cols_per_node, 0.0);
  std::vector<double> ciNodesV(2 * constrainsNo * cols_per_node, 0.0);

  double max_norm_Gu = 0.0, max_norm_Gv = 0.0;
  double max_abs_Gu = 0.0, max_abs_Gv = 0.0;
  double sum_norm_G = 0.0;
  int n_nodes_failed = 0;
  std::vector<bool> node_failed(constrainsNo, false);

  // Helper: compute max Frobenius norm of G_u and G_v
  auto compute_G_norm = [&](const std::vector<double> &gu0,
                            const std::vector<double> &gu1,
                            const std::vector<double> &gv0,
                            const std::vector<double> &gv1) -> double {
    int cpn = 2 * ABCS_STEP * maxCloudSize;
    double sq_u = 0.0, sq_v = 0.0;
    for (int c = 0; c < cpn; c++) {
      sq_u += gu0[c] * gu0[c] + gu1[c] * gu1[c];
      sq_v += gv0[c] * gv0[c] + gv1[c] * gv1[c];
    }
    return std::max(std::sqrt(sq_u), std::sqrt(sq_v));
  };

  // Helper: store G into output arrays
  auto store_G = [&](int i,
                     const std::vector<double> &gu0, const std::vector<double> &gu1,
                     const std::vector<double> &gv0, const std::vector<double> &gv1) {
    int row0 = 2 * i, row1 = 2 * i + 1;
    for (int c = 0; c < cols_per_node; c++) {
      ciNodesU[row0 * cols_per_node + c] = gu0[c];
      ciNodesU[row1 * cols_per_node + c] = gu1[c];
      ciNodesV[row0 * cols_per_node + c] = gv0[c];
      ciNodesV[row1 * cols_per_node + c] = gv1[c];
    }
    int cpn = 2 * ABCS_STEP * maxCloudSize;
    double sq_u = 0.0, sq_v = 0.0;
    double node_max_gu = 0.0, node_max_gv = 0.0;
    for (int c = 0; c < cpn; c++) {
      sq_u += gu0[c] * gu0[c] + gu1[c] * gu1[c];
      sq_v += gv0[c] * gv0[c] + gv1[c] * gv1[c];
      node_max_gu = std::max(node_max_gu, std::max(std::abs(gu0[c]), std::abs(gu1[c])));
      node_max_gv = std::max(node_max_gv, std::max(std::abs(gv0[c]), std::abs(gv1[c])));
    }
    max_norm_Gu = std::max(max_norm_Gu, std::sqrt(sq_u));
    max_norm_Gv = std::max(max_norm_Gv, std::sqrt(sq_v));
    max_abs_Gu = std::max(max_abs_Gu, node_max_gu);
    max_abs_Gv = std::max(max_abs_Gv, node_max_gv);
  };

  // =========================================================================
  // Global mode count: ALL modes used for ALL nodes (matching BB-PD Mathematica).
  // The system is always underdetermined (nBasis=224 > nDataRows=196 for full
  // cloud). LeastSquares/dgelsd returns minimum-norm solution.
  // In BB-PD, this naturally gives ||G|| ≈ 1. No rcond tuning needed.
  // =========================================================================
  int nDataRows_typical = 2 * maxCloudSize * ABCS_STEP;
  int nModesGlobal = static_cast<int>(modes.size());  // use ALL modes
  int nBasisGlobal = 4 * nModesGlobal;

  std::cout << std::endl << "=== Global Mode Count (BB-PD style) ===" << std::endl;
  std::cout << "  SVD rcond = " << SVD_RCOND << std::endl;
  std::cout << "  nDataRows (typical, cloud=" << maxCloudSize << ") = " << nDataRows_typical << std::endl;
  std::cout << "  nModesGlobal = " << nModesGlobal << " (of " << modes.size() << " available)" << std::endl;
  std::cout << "  nBasisGlobal = " << nBasisGlobal << std::endl;
  std::cout << "  ratio nBasis/nDataRows = " << (double)nBasisGlobal / nDataRows_typical << std::endl;

  // =========================================================================
  // 2D SWEEP: find (dt, delta_kappa) that minimizes max||G|| across all nodes.
  // delta_kappa changes the mode set; dt changes the time factors.
  // Sample the most problematic nodes (smallest clouds + corners).
  // =========================================================================
  double dt_optimal = DT;
  double dk_optimal = delta_kappa;
  {
    // Build sample of problematic nodes
    std::vector<int> sample_nodes;
    for (int i = 0; i < constrainsNo; i++) {
      int cs = static_cast<int>(clouds[i].size());
      if (cs <= 26) sample_nodes.push_back(i);  // all truncated nodes
    }
    for (int i = 0; i < std::min(20, constrainsNo); i++) sample_nodes.push_back(i);
    for (int i = std::max(0, constrainsNo - 20); i < constrainsNo; i++) sample_nodes.push_back(i);
    std::sort(sample_nodes.begin(), sample_nodes.end());
    sample_nodes.erase(std::unique(sample_nodes.begin(), sample_nodes.end()), sample_nodes.end());
    std::cout << std::endl << "=== 2D Sweep (dt, delta_kappa) ===" << std::endl;
    std::cout << "  Sampling " << sample_nodes.size() << " nodes" << std::endl;

    // Grid: dt from 5ns to 35ns, delta_kappa from 0.5x to 2.0x of paper value
    double dk_base = delta_kappa;
    double dk_factors[] = {0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0};
    double dt_values[] = {5.0e-9, 8.0e-9, 10.0e-9, 12.0e-9, 14.0e-9, 15.0e-9,
                          16.0e-9, 17.0e-9, 17.5e-9, 18.0e-9, 20.0e-9, 25.0e-9, 30.0e-9, 35.0e-9};
    int n_dk = sizeof(dk_factors) / sizeof(dk_factors[0]);
    int n_dt = sizeof(dt_values) / sizeof(dt_values[0]);

    // Store all results for ranking
    struct SweepResult {
      double dk_factor, dk, dt, maxG, avg_cond;
      int nModes;
      bool stable;
    };
    std::vector<SweepResult> results;

    std::cout << std::setw(10) << "dk_fac" << std::setw(8) << "nMode"
              << std::setw(10) << "dt(ns)" << std::setw(12) << "max||G||"
              << std::setw(12) << "avg_cond" << std::setw(10) << "status" << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    for (int idk = 0; idk < n_dk; idk++) {
      double dk_try = dk_base * dk_factors[idk];
      std::vector<WaveMode> modes_try = compute_all_modes(mc, tempGrid, dk_try);
      int nModes_try = static_cast<int>(modes_try.size());

      std::sort(modes_try.begin(), modes_try.end(), [](const WaveMode &a, const WaveMode &b) {
        return (a.alpha1*a.alpha1 + a.alpha2*a.alpha2) < (b.alpha1*b.alpha1 + b.alpha2*b.alpha2);
      });

      for (int idt = 0; idt < n_dt; idt++) {
        double dt_try = dt_values[idt];
        double max_G = 0.0;
        double sum_cond = 0.0;
        int n_cond = 0;

        for (int idx : sample_nodes) {
          std::vector<double> gu0, gu1, gv0, gv1;
          compute_abc_node(idx, nodes_x[abc_indices[idx]], nodes_y[abc_indices[idx]],
                           abc_tau[idx], clouds[idx], nodes_x, nodes_y,
                           modes_try, mc, maxCloudSize, gu0, gu1, gv0, gv1,
                           -99.0, nModes_try, dt_try);
          int cpn = 2 * ABCS_STEP * maxCloudSize;
          double sq_u = 0.0, sq_v = 0.0;
          for (int c = 0; c < cpn; c++) {
            sq_u += gu0[c] * gu0[c] + gu1[c] * gu1[c];
            sq_v += gv0[c] * gv0[c] + gv1[c] * gv1[c];
          }
          double node_G = std::max(std::sqrt(sq_u), std::sqrt(sq_v));
          max_G = std::max(max_G, node_G);
          if (max_G > 1.5) break;  // early exit
        }

        SweepResult r;
        r.dk_factor = dk_factors[idk];
        r.dk = dk_try;
        r.dt = dt_try;
        r.maxG = max_G;
        r.avg_cond = (n_cond > 0) ? sum_cond / n_cond : 0;
        r.nModes = nModes_try;
        r.stable = (max_G <= 1.0);
        results.push_back(r);

        // Print stable results and near-stable
        if (max_G <= 1.02) {
          std::cout << std::setw(10) << dk_factors[idk] << std::setw(8) << nModes_try
                    << std::setw(10) << dt_try * 1e9 << std::setw(12) << max_G
                    << std::setw(12) << "-"
                    << std::setw(10) << (max_G <= 1.0 ? "STABLE" : "marginal") << std::endl;
        }
      }
    }

    // Among stable results, pick the one with max||G|| closest to 1 (best absorption)
    double best_score = -1.0;
    double best_dt = DT, best_dk = dk_base;
    int best_nModes = 0;
    double best_maxG = 0.0;
    for (const auto &r : results) {
      if (r.stable && r.maxG > best_score) {
        best_score = r.maxG;
        best_dt = r.dt;
        best_dk = r.dk;
        best_nModes = r.nModes;
        best_maxG = r.maxG;
      }
    }
    if (best_score < 0) {
      // No stable combination found — use the one with smallest max||G||
      double min_G = 1e10;
      for (const auto &r : results) {
        if (r.maxG < min_G) { min_G = r.maxG; best_dt = r.dt; best_dk = r.dk; best_nModes = r.nModes; best_maxG = r.maxG; }
      }
      std::cout << "  WARNING: No stable combination found! Closest: max||G||=" << min_G << std::endl;
    }

    std::cout << std::endl << ">>> Best stable: dk_factor=" << best_dk / dk_base
              << " dk=" << best_dk << " dt=" << best_dt * 1e9 << "ns"
              << " nModes=" << best_nModes << " max||G||=" << best_maxG << std::endl;

    dt_optimal = best_dt;
    dk_optimal = best_dk;

    // Recompute modes with optimal dk
    modes = compute_all_modes(mc, tempGrid, dk_optimal);
    nModesGlobal = static_cast<int>(modes.size());
    nBasisGlobal = 4 * nModesGlobal;
    std::sort(modes.begin(), modes.end(), [](const WaveMode &a, const WaveMode &b) {
      return (a.alpha1*a.alpha1 + a.alpha2*a.alpha2) < (b.alpha1*b.alpha1 + b.alpha2*b.alpha2);
    });
    std::cout << "  Final mode set: nModes=" << nModesGlobal << " nBasis=" << nBasisGlobal << std::endl;
  }

  // Compute N_STEPS for same physical simulation time as original
  double T_final = 6000.0 * 35.0e-9;  // original: 6000 steps * 35 ns = 210 µs
  int n_steps_opt = static_cast<int>(std::ceil(T_final / dt_optimal));
  std::cout << "  N_STEPS = " << n_steps_opt << " (T_final = " << T_final*1e6 << " µs)" << std::endl;

  std::cout << std::endl << "Computing ABC operators for " << constrainsNo
            << " nodes with dt=" << dt_optimal*1e9 << " ns..." << std::endl;

  for (int i = 0; i < constrainsNo; i++) {
    int nCloud = static_cast<int>(clouds[i].size());
    int nDataRows_i = 2 * nCloud * ABCS_STEP;

    std::vector<double> gu0, gu1, gv0, gv1;
    // Use ALL modes, rcond=-1 (machine epsilon) — matching BB-PD Mathematica exactly
    compute_abc_node(i, nodes_x[abc_indices[i]], nodes_y[abc_indices[i]],
                     abc_tau[i], clouds[i], nodes_x, nodes_y,
                     modes, mc, maxCloudSize, gu0, gu1, gv0, gv1,
                     -99.0, nModesGlobal, dt_optimal);
    double norm = compute_G_norm(gu0, gu1, gv0, gv1);

    if (norm > G_NORM_TARGET) {
      n_nodes_failed++;
      node_failed[i] = true;
      std::cout << "  [||G||>1] node " << i << " ||G||=" << norm
                << " cloud=" << nCloud << " nBasis=" << nBasisGlobal
                << " nData=" << nDataRows_i << std::endl;
    }
    store_G(i, gu0, gu1, gv0, gv1);
    sum_norm_G += norm;

    if ((i + 1) % 500 == 0 || i == constrainsNo - 1) {
      std::cout << "  " << (i + 1) << " / " << constrainsNo
                << "  (nBasis=" << nBasisGlobal << "/" << nDataRows_i
                << ", ||G||=" << norm
                << ", cloud=" << nCloud << ")" << std::endl;
    }
  }

  std::cout << std::endl << "=== ABC Operator Diagnostics ===" << std::endl;
  std::cout << "  Global modes = " << nModesGlobal << ", nBasis = " << nBasisGlobal << std::endl;
  std::cout << "  max ||G_u||_F = " << max_norm_Gu << std::endl;
  std::cout << "  max ||G_v||_F = " << max_norm_Gv << std::endl;
  std::cout << "  max |G_u_ij| = " << max_abs_Gu << std::endl;
  std::cout << "  max |G_v_ij| = " << max_abs_Gv << std::endl;
  std::cout << "  avg ||G||    = " << sum_norm_G / constrainsNo << std::endl;
  std::cout << "  Nodes FAILED (||G|| > " << G_NORM_TARGET << "): " << n_nodes_failed
            << " / " << constrainsNo << std::endl;

  // ========================================================================
  // 7b. Compute G_theta for all absorbing nodes (Sec. 4.8)
  // ========================================================================
  int cols_theta_per_node = ABCS_STEP * maxCloudSize;
  std::vector<double> ciNodesTheta(constrainsNo * cols_theta_per_node, 0.0);
  double max_norm_Gtheta = 0.0, max_abs_Gtheta = 0.0;

  std::cout << std::endl << "Computing G_theta operators..." << std::endl;

  for (int i = 0; i < constrainsNo; i++) {
    std::vector<double> g_theta;
    compute_abc_node_theta(i, nodes_x[abc_indices[i]], nodes_y[abc_indices[i]],
                            abc_tau[i], clouds[i], nodes_x, nodes_y,
                            modes, mc, maxCloudSize, g_theta,
                            -99.0, nModesGlobal, dt_optimal);

    double sq = 0.0;
    for (int c = 0; c < cols_theta_per_node; c++) {
      ciNodesTheta[i * cols_theta_per_node + c] = g_theta[c];
      sq += g_theta[c] * g_theta[c];
      max_abs_Gtheta = std::max(max_abs_Gtheta, std::abs(g_theta[c]));
    }
    double norm = std::sqrt(sq);
    max_norm_Gtheta = std::max(max_norm_Gtheta, norm);

    if ((i + 1) % 500 == 0 || i == constrainsNo - 1) {
      std::cout << "  G_theta: " << (i + 1) << " / " << constrainsNo
                << " ||G_theta||=" << norm << std::endl;
    }
  }

  std::cout << std::endl << "=== G_theta Diagnostics ===" << std::endl;
  std::cout << "  max ||G_theta||_F = " << max_norm_Gtheta << std::endl;
  std::cout << "  max |G_theta_ij| = " << max_abs_Gtheta << std::endl;

  // ========================================================================
  // 8. Use ALL ABC nodes (matching BB-PD Mathematica — no exclusions)
  // ========================================================================
  std::vector<int> active_abc;
  for (int i = 0; i < constrainsNo; i++) {
    active_abc.push_back(i);  // include ALL nodes, regardless of ||G||
  }
  int nActive = static_cast<int>(active_abc.size());
  std::cout << std::endl << "Active ABC nodes: " << nActive << " / " << constrainsNo
            << " (ALL included, BB-PD style)" << std::endl;

  // Set up boundary conditions
  std::vector<int> bcTypes(2 * NodeNo, -1);
  std::vector<double> bcValues(2 * NodeNo, 0.0);
  for (int a = 0; a < nActive; a++) {
    int n = abc_indices[active_abc[a]];
    bcTypes[2 * n]     = 1;
    bcTypes[2 * n + 1] = 1;
  }

  // Gaussian pulse IC (Example 1)
  std::vector<double> icValues(2 * NodeNo, 0.0);
  std::vector<double> icdValues(2 * NodeNo, 0.0);
  for (int i = 0; i < NodeNo; i++) {
    double r2 = nodes_x[i] * nodes_x[i] + nodes_y[i] * nodes_y[i];
    if (std::sqrt(r2) < 40.0) {  // r < 40 mm (matching BB-PD Mathematica)
      double val = std::exp(-r2 / 100.0);
      icValues[2 * i]     = val;
      icValues[2 * i + 1] = val;
    }
  }

  // ========================================================================
  // 9. Export data files (only active ABC nodes)
  // ========================================================================
  std::cout << std::endl << "Exporting data files..." << std::endl;

  // .data1 — nActive replaces constrainsNo
  {
    std::ofstream f(jobname + ".data1");
    f << NodeNo << std::endl;
    f << n_steps_opt << std::endl;
    f << 0 << std::endl;           // PrenotchNo
    f << 0 << std::endl;           // VINodesNo
    f << nActive << std::endl;     // only active ABC nodes
    f << maxCloudSize << std::endl;
  }

  // .data2 (extended 15-entry format)
  {
    std::ofstream f(jobname + ".data2");
    f << std::setprecision(16);
    f << mc.E << std::endl;
    f << mc.nu << std::endl;
    f << mc.rho << std::endl;
    f << 0.0 << std::endl;             // wf
    f << 0.0 << std::endl;             // t0
    f << n_steps_opt * dt_optimal << std::endl;    // t1
    f << dt_optimal << std::endl;
    f << DELTA << std::endl;
    f << DX << std::endl;
    f << DY << std::endl;
    f << mc.s0 << std::endl;
    f << mc.m_ref << std::endl;        // index 12
    f << mc.kappa << std::endl;        // index 13
    f << mc.alpha_lps << std::endl;    // index 14
    f << mc.lambda_pd << std::endl;    // index 15
  }

  // .nodes
  {
    std::ofstream f(jobname + ".nodes");
    f << std::setprecision(16);
    for (int i = 0; i < NodeNo; i++) {
      f << nodes_x[i] << " " << nodes_y[i] << std::endl;
    }
  }

  // .bctypes
  {
    std::ofstream f(jobname + ".bctypes");
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << bcTypes[i] << std::endl;
    }
  }

  // .bcvalues
  {
    std::ofstream f(jobname + ".bcvalues");
    f << std::setprecision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << bcValues[i] << std::endl;
    }
  }

  // .icvalues
  {
    std::ofstream f(jobname + ".icvalues");
    f << std::setprecision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << icValues[i] << std::endl;
    }
  }

  // .icdvalues
  {
    std::ofstream f(jobname + ".icdvalues");
    f << std::setprecision(16);
    for (int i = 0; i < 2 * NodeNo; i++) {
      f << icdValues[i] << std::endl;
    }
  }

  // .prenotch (empty)
  {
    std::ofstream f(jobname + ".prenotch");
  }

  // .vinodes (empty)
  {
    std::ofstream f(jobname + ".vinodes");
  }

  // .infNodesIndex — only active ABC node indices
  {
    std::ofstream f(jobname + ".infNodesIndex");
    for (int a = 0; a < nActive; a++) {
      f << abc_indices[active_abc[a]] << std::endl;
    }
  }

  // .iClouds (nActive x maxCloudSize, padded with -1) — only active nodes
  {
    std::ofstream f(jobname + ".iClouds");
    for (int a = 0; a < nActive; a++) {
      int i = active_abc[a];
      for (int j = 0; j < maxCloudSize; j++) {
        if (j > 0) f << " ";
        if (j < static_cast<int>(clouds[i].size()))
          f << clouds[i][j];
        else
          f << -1;
      }
      f << std::endl;
    }
  }

  // .ciNodesU (2*nActive x cols_per_node) — only active nodes
  {
    std::ofstream f(jobname + ".ciNodesU");
    f << std::setprecision(16);
    for (int a = 0; a < nActive; a++) {
      int i = active_abc[a];
      // Row 0 (x-component) and Row 1 (y-component)
      for (int r = 0; r < 2; r++) {
        int src_row = 2 * i + r;
        for (int j = 0; j < cols_per_node; j++) {
          if (j > 0) f << " ";
          f << ciNodesU[src_row * cols_per_node + j];
        }
        f << std::endl;
      }
    }
  }

  // .ciNodesV (2*nActive x cols_per_node) — only active nodes
  {
    std::ofstream f(jobname + ".ciNodesV");
    f << std::setprecision(16);
    for (int a = 0; a < nActive; a++) {
      int i = active_abc[a];
      for (int r = 0; r < 2; r++) {
        int src_row = 2 * i + r;
        for (int j = 0; j < cols_per_node; j++) {
          if (j > 0) f << " ";
          f << ciNodesV[src_row * cols_per_node + j];
        }
        f << std::endl;
      }
    }
  }

  // .ciNodesTheta (nActive x cols_theta_per_node) — G_theta updating vector
  {
    std::ofstream f(jobname + ".ciNodesTheta");
    f << std::setprecision(16);
    for (int a = 0; a < nActive; a++) {
      int i = active_abc[a];
      for (int j = 0; j < cols_theta_per_node; j++) {
        if (j > 0) f << " ";
        f << ciNodesTheta[i * cols_theta_per_node + j];
      }
      f << std::endl;
    }
  }

  std::cout << std::endl << "All files written to " << jobname << ".*" << std::endl;
  std::cout << "  constrainsNo (active) = " << nActive << std::endl;
  std::cout << "  maxCloudSize = " << maxCloudSize << std::endl;
  std::cout << "Run solver: ./PD2_LPS_ABC _out " << jobname << std::endl;

  return 0;
}
