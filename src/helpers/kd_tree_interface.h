#include <algorithm>
#include <cmath>
#include <kdtree++/kdtree.hpp>
#include <stdexcept>

typedef void (*build_cloud_callback)(size_t i, size_t &min_node_node, double &r,
                                     double &dr);

class Node;

Node *ref_node;

class Node {
 public:
  typedef double value_type;

  double xy[2];
  int index;

  double operator[](int n) const { return xy[n]; }

  double distance(const Node &node) const {
    double x = xy[0] - node.xy[0];
    double y = xy[1] - node.xy[1];

    return std::max(std::abs(x), std::abs(y));
  }

  double distance_euclidean(const Node &node) const {
    double x = xy[0] - node.xy[0];
    double y = xy[1] - node.xy[1];

    return std::sqrt(x * x + y * y);
  }
};

bool NodeComp(const Node &i, const Node &j) {
  return i.distance_euclidean(*ref_node) < j.distance_euclidean(*ref_node);
}

template <typename double_matrix_type, typename int_matrix_type,
          typename int_vector_type>
void build_clouds(int min_node_no, int max_node_no, double r, double dr,
                  double_matrix_type &nodes, int_matrix_type &clouds,
                  int_vector_type &cloud_lengths, int &max_cloud_length) {
  KDTree::KDTree<2, Node> tree;

  ref_node = new Node;

  int size = nodes.size1();

  clouds.resize(size, max_node_no);
  cloud_lengths.resize(size);

  for (int i = 0; i < size; i++) {
    Node node;

    node.xy[0] = nodes(i, 0);
    node.xy[1] = nodes(i, 1);

    node.index = i;

    tree.insert(node);
  }

  tree.optimize();

  std::vector<Node> found_nodes;

  max_cloud_length = 0;

  for (int i = 0; i < size; i++) {
    ref_node->xy[0] = nodes(i, 0);
    ref_node->xy[1] = nodes(i, 1);

    double current_r = r;

    found_nodes.reserve(max_node_no);
    int found_node_no = 0;

    do {
      found_nodes.clear();

      tree.find_within_range(
          *ref_node, current_r,
          std::back_insert_iterator<std::vector<Node> >(found_nodes));

      found_node_no = 0;

      for (int j = 0; j < static_cast<int>(found_nodes.size()); j++) {
        if (found_nodes[j].distance_euclidean(*ref_node) <= current_r) {
          found_node_no++;
        }
      }

      if (found_node_no > max_node_no) {
        throw std::runtime_error(
            "found_node_no > max_node_no; increase max_node_no.");
      }

      if (found_node_no < min_node_no) {
        current_r += dr;
      }

    } while (found_node_no < min_node_no);

    std::sort(found_nodes.begin(), found_nodes.end(), NodeComp);

    int current_node = 0;

    for (int j = 0; j < static_cast<int>(found_nodes.size()); j++) {
      if (found_nodes[j].distance_euclidean(*ref_node) <= current_r) {
        clouds(i, current_node++) = found_nodes[j].index;
      }
    }

    for (int j = found_node_no; j < max_node_no; j++) {
      clouds(i, j) = -1;
    }

    // std::cout << found_node_no << std::endl;

    cloud_lengths[i] = found_node_no;
    max_cloud_length = std::max(max_cloud_length, found_node_no);
  }
}

template <typename double_matrix_type, typename int_matrix_type,
          typename int_vector_type>
void build_clouds_adaptive(build_cloud_callback cloud_callback,
                           size_t max_node_no, double_matrix_type &nodes,
                           int_matrix_type &clouds,
                           int_vector_type &cloud_lengths,
                           size_t &max_cloud_length) {
  KDTree::KDTree<2, Node> tree;

  ref_node = new Node;

  size_t size = nodes.size1();

  clouds.resize(size, max_node_no);
  cloud_lengths.resize(size);

  for (size_t i = 0; i < size; i++) {
    Node node;

    node.xy[0] = nodes(i, 0);
    node.xy[1] = nodes(i, 1);

    node.index = i;

    tree.insert(node);
  }

  tree.optimize();

  std::vector<Node> found_nodes;

  max_cloud_length = 0;

  for (size_t i = 0; i < size; i++) {
    ref_node->xy[0] = nodes(i, 0);
    ref_node->xy[1] = nodes(i, 1);

    size_t min_node_no;
    double r;
    double dr;

    (*cloud_callback)(i, min_node_no, r, dr);

    double current_r = r;

    found_nodes.reserve(max_node_no);
    size_t found_node_no = 0;

    do {
      found_nodes.clear();

      tree.find_within_range(
          *ref_node, current_r,
          std::back_insert_iterator<std::vector<Node> >(found_nodes));

      found_node_no = 0;

      for (size_t j = 0; j < found_nodes.size(); j++) {
        if (found_nodes[j].distance_euclidean(*ref_node) <= current_r) {
          found_node_no++;
        }
      }

      if (found_node_no > max_node_no) {
        throw std::runtime_error(
            "found_node_no > max_node_no; increase max_node_no.");
      }

      if (found_node_no < min_node_no) {
        current_r += dr;
      }

    } while (found_node_no < min_node_no);

    std::sort(found_nodes.begin(), found_nodes.end(), NodeComp);

    size_t current_node = 0;

    for (size_t j = 0; j < found_nodes.size(); j++) {
      if (found_nodes[j].distance_euclidean(*ref_node) <= current_r) {
        clouds(i, current_node++) = found_nodes[j].index;
      }
    }

    for (size_t j = found_node_no; j < max_node_no; j++) {
      clouds(i, j) = -1;
    }

    // std::cout << found_node_no << std::endl;

    cloud_lengths[i] = found_node_no;
    max_cloud_length = std::max(max_cloud_length, found_node_no);
  }
}
