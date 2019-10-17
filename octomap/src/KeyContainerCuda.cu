#ifdef __CUDA_SUPPORT__
#include <octomap/KeyArrayCuda.cuh>

namespace octomap {
  __device__ void KeyArrayCuda::addKeyAtomic(const OcTreeKey& k) {
      int idx = atomicAdd(&last, 1);
      ray[idx] = k;
    }

  __device__ void KeyArray3DCuda::setAtomic(const int& val, const size_t& r, const size_t& c, const size_t& d) {
      atomicMax(&arr[c + r * maxRowSize + d * maxRowSize2], val); // unknown to free, free to occupied -- > -1, 0, 1
  }
}
#endif