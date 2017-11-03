#ifndef LIB_TEST_TEST_UTIL_H_
#define LIB_TEST_TEST_UTIL_H_

#include <iostream>
#include "../grid_mpi.h"
#include "../array.h"

template <class T> inline
std::ostream& print_grid(GridMPI *g, int my_rank, std::ostream &os) {
  T *data = (T*)g->data();
  T **halo_self_fw = (T**)g->halo_self_fw();
  T **halo_self_bw = (T**)g->halo_self_bw();
  T **halo_peer_fw = (T**)g->halo_peer_fw();
  T **halo_peer_bw = (T**)g->halo_peer_bw();
  IndexArray lsize = g->my_size();
  std::stringstream ss;
  ss << "[rank:" << my_rank << "] ";

  StringJoin sj;
  for (ssize_t i = 0; i < lsize.accumulate(g->num_dims()); ++i) {
      sj << data[i];
    }
    ss << "data {" << sj << "}";
  ss << "\n";
  os << ss.str();;
  return os;
}


#endif
