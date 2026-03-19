#include <fstream>
#include <cmath>
#include <string>
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#endif

namespace utils {

template <typename vector_type>
void calc_mag(vector_type &mag, vector_type &vec) {
  for (int i = 0; i < vec.size() / 2; i++) {
    mag[i] =
        std::sqrt(vec[2 * i] * vec[2 * i] + vec[2 * i + 1] * vec[2 * i + 1]);
  }
}

template <typename vector_type>
bool read_vector_from_file(const std::string &filename, vector_type &vec,
                           int size) {
  std::ifstream file(filename.c_str());
  if (!file) return false;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    file >> vec[i];
  }
  return true;
}

template <typename matrix_type>
bool read_matrix_from_file(const std::string &filename, matrix_type &mat,
                           int size1, int size2) {
  std::ifstream file(filename.c_str());
  if (!file) return false;
  mat.resize(size1, size2);
  for (int i = 0; i < size1; i++) {
    for (int j = 0; j < size2; j++) {
      file >> mat(i, j);
    }
  }
  return true;
}

// Read integer index arrays from files that may contain floating-point formatted values
// (e.g., Mathematica exports "42.0" instead of "42"). Reads as double, casts to int.
template <typename int_vector_type>
bool read_int_vector_from_file(const std::string &filename, int_vector_type &vec,
                               int size) {
  std::ifstream file(filename.c_str());
  if (!file) return false;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    double tmp;
    file >> tmp;
    vec[i] = static_cast<int>(tmp + 0.5);  // round to nearest int
  }
  return true;
}

template <typename int_matrix_type>
bool read_int_matrix_from_file(const std::string &filename, int_matrix_type &mat,
                               int size1, int size2) {
  std::ifstream file(filename.c_str());
  if (!file) return false;
  mat.resize(size1, size2);
  for (int i = 0; i < size1; i++) {
    for (int j = 0; j < size2; j++) {
      double tmp;
      file >> tmp;
      mat(i, j) = static_cast<int>(tmp + 0.5);
    }
  }
  return true;
}

template <typename vector_type>
bool write_vector_to_file(const std::string &filename, const vector_type &vec) {
  std::ofstream file(filename.c_str());
  if (!file) return false;
  file.precision(16);
  for (int i = 0; i < vec.size(); i++) {
    file << vec[i] << std::endl;
  }
  return true;
}

// VTK Legacy output for 2D point data with vector and scalar fields
template <typename vector_type, typename matrix_type>
bool write_vtk_vector(const std::string &filename,
                      const matrix_type &nodes,
                      const vector_type &vec,
                      const char *vec_name,
                      int node_count,
                      double exclude_radius = -1.0) {
  // First pass: count points to write
  int count = 0;
  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    count++;
  }

  std::ofstream file(filename.c_str());
  if (!file) return false;
  file.precision(12);

  file << "# vtk DataFile Version 3.0" << std::endl;
  file << "PD2_LPS_ABC output" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET POLYDATA" << std::endl;
  file << "POINTS " << count << " double" << std::endl;

  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    file << nodes(i, 0) << " " << nodes(i, 1) << " 0.0" << std::endl;
  }

  // Vertices connectivity
  file << "VERTICES " << count << " " << 2 * count << std::endl;
  for (int i = 0; i < count; i++) {
    file << "1 " << i << std::endl;
  }

  file << "POINT_DATA " << count << std::endl;

  // Vector field (displacement or velocity)
  file << "VECTORS " << vec_name << " double" << std::endl;
  int idx = 0;
  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    file << vec[2 * i] << " " << vec[2 * i + 1] << " 0.0" << std::endl;
    idx++;
  }

  // Scalar magnitude
  file << "SCALARS " << vec_name << "_magnitude double 1" << std::endl;
  file << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    double mag = std::sqrt(vec[2 * i] * vec[2 * i] +
                           vec[2 * i + 1] * vec[2 * i + 1]);
    file << mag << std::endl;
  }

  return true;
}

// VTK scalar field output
template <typename vector_type, typename matrix_type>
bool write_vtk_scalar(const std::string &filename,
                      const matrix_type &nodes,
                      const vector_type &vec,
                      const char *scalar_name,
                      int node_count,
                      double exclude_radius = -1.0) {
  int count = 0;
  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    count++;
  }

  std::ofstream file(filename.c_str());
  if (!file) return false;
  file.precision(12);

  file << "# vtk DataFile Version 3.0" << std::endl;
  file << "PD2_LPS_ABC output" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET POLYDATA" << std::endl;
  file << "POINTS " << count << " double" << std::endl;

  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    file << nodes(i, 0) << " " << nodes(i, 1) << " 0.0" << std::endl;
  }

  file << "VERTICES " << count << " " << 2 * count << std::endl;
  for (int i = 0; i < count; i++) {
    file << "1 " << i << std::endl;
  }

  file << "POINT_DATA " << count << std::endl;
  file << "SCALARS " << scalar_name << " double 1" << std::endl;
  file << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < node_count; i++) {
    if (exclude_radius > 0.0) {
      double r = std::sqrt(nodes(i, 0) * nodes(i, 0) +
                           nodes(i, 1) * nodes(i, 1));
      if (r > exclude_radius) continue;
    }
    file << vec[i] << std::endl;
  }

  return true;
}

// Legacy CSV output (kept for compatibility)
template <typename vector_type, typename matrix_type>
bool write_vector_paraview_vector_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec,
                                          const char *col_name_x,
                                          const char *col_name_y) {
  std::ofstream file(filename.c_str());
  if (!file) return false;
  file.precision(16);
  file << "X,Y,Z," << col_name_x << "," << col_name_y << std::endl;
  for (int i = 0; i < vec.size() / 2; i++) {
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << 0.0;
    for (int j = 0; j < 2; j++) {
      file << "," << vec[2 * i + j];
    }
    file << std::endl;
  }
  return true;
}

template <typename vector_type, typename matrix_type>
bool write_vector_paraview_scalar_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec,
                                          const char *col_name) {
  std::ofstream file(filename.c_str());
  if (!file) return false;
  file.precision(16);
  file << "X,Y,Z," << col_name << std::endl;
  for (int i = 0; i < vec.size(); i++) {
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << 0 << "," << vec[i]
         << std::endl;
  }
  return true;
}

class timer {
 public:
  timer() {}

  void tic() {
#ifdef _WIN32
    QueryPerformanceCounter(&start);
#else
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
  }

  double toc() {
#ifdef _WIN32
    LARGE_INTEGER stop;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&stop);
    QueryPerformanceFrequency(&frequency);
    return (stop.QuadPart - start.QuadPart) /
           static_cast<double>(frequency.QuadPart);
#else
    struct timespec stop;
    clock_gettime(CLOCK_MONOTONIC, &stop);
    return static_cast<double>(((unsigned long long)stop.tv_sec *
                                    (1000ULL * 1000ULL * 1000ULL) +
                                (unsigned long long)stop.tv_nsec) -
                               ((unsigned long long)start.tv_sec *
                                    (1000ULL * 1000ULL * 1000ULL) +
                                (unsigned long long)start.tv_nsec)) /
           1.00e09;
#endif
  }

 private:
#ifdef _WIN32
  LARGE_INTEGER start;
#else
  struct timespec start;
#endif
};

}  // namespace utils
