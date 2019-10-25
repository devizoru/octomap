#ifndef KEY_ARRAY_CUDA_CUH
#define KEY_ARRAY_CUDA_CUH
#ifdef __CUDA_SUPPORT__
#include <cuda.h>
#include <cuda_runtime.h>
#include <octomap/AssertionCuda.cuh>
#include <octomap/OcTreeKey.h>

namespace octomap {

  #define UNKNOWN -1
  #define FREE 0
  #define OCCUPIED 1

  struct KeyRayConfig {
    static int max_ray_size_;
    static void setMaxSize(const int& max_size) { max_ray_size_ = max_size;} 
  };

  struct KeyHash {
    __host__ __device__ KeyHash() {}
    __host__ __device__ KeyHash(const OcTreeKey& key, const size_t& hash) :
      key_(key), hash_(hash) {}
    __host__ __device__ ~KeyHash() {}
    OcTreeKey key_;
    size_t hash_;
  };

  template <typename T> 
  class ArrayCuda {
  public:
    
    __host__ ArrayCuda (const int& max_size = KeyRayConfig::max_ray_size_) :
      max_size_(max_size)
    {      
      reset();
    }

    __host__ ~ArrayCuda () {
    }
    
    CUDA_CALLABLE ArrayCuda(const ArrayCuda& other)
    {
      ray_ = other.ray_;
      last_ = other.last_;
    }
    
    __host__ void allocateHost() {
      ray_ = (T*)malloc(max_size_ * sizeof(T));
    }

    __host__ void allocateDevice() {
      cudaCheckErrors(cudaMallocManaged(&ray_, max_size_ * sizeof(T)))
    }

    __host__ void freeDevice() {
      cudaCheckErrors(cudaFree(ray_));
    }

    __host__ void copyFromHost(const ArrayCuda& other) {
      cudaCheckErrors(cudaMemcpy(ray_, other.ray_, max_size_ * sizeof(T), cudaMemcpyHostToDevice));
      last_ = other.last_;
    }

    __host__ void copyFromDevice(const ArrayCuda& other) {
      cudaCheckErrors(cudaMemcpy(ray_, other.ray_, max_size_ * sizeof(T), cudaMemcpyDeviceToHost));
      last_ = other.last_;
    }

    CUDA_CALLABLE ArrayCuda& operator=(const ArrayCuda& other){
      ray_ = other.ray_;
      last_ = other.last_;
      return *this;
    }
    
    CUDA_CALLABLE void reset() {
      last_ = 0;
    }
    
    CUDA_CALLABLE void addKey(const T& k) {
      ray_[last_] = k;
      ++last_;
    }

    CUDA_CALLABLE void addKey(const T& k, const int& l) {
      ray_[l] = k;
    }

    CUDA_CALLABLE T& operator[](int i) { 
      return ray_[i];
    }

    CUDA_CALLABLE T operator[](int i) const { 
      return ray_[i];
    }
    
    CUDA_CALLABLE T& get(const int& i) { 
      return ray_[i];
    }

    CUDA_CALLABLE T get(const int& i) const { 
      return ray_[i];
    }

    __device__ void addKeyAtomic(const T& k);

    CUDA_CALLABLE T* begin() { return ray_; }
    CUDA_CALLABLE T* end() { return &ray_[last_]; }

    CUDA_CALLABLE int size() const { return last_; }
    CUDA_CALLABLE int* last() { return &last_; }
    int sizeMax() { return max_size_; }

  protected:
    T* ray_;
    int last_;
    int max_size_;
  };

  class UnsignedArrayCuda : public ArrayCuda<unsigned> {
  public:
    __device__ unsigned addKeyAtomicAt(const unsigned& k, const unsigned& l);
    using ArrayCuda::ArrayCuda;
  };

  class KeyArrayCuda : public ArrayCuda<OcTreeKey> {
    using ArrayCuda::ArrayCuda;
  };

  using KeyRayCuda = KeyArrayCuda;
}


#endif
#endif