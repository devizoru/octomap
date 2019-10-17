#ifndef KEY_CONTAINDER_CUDA_CUH
#define KEY_CONTAINDER_CUDA_CUH
#ifdef __CUDA_SUPPORT__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi.h>
#include <octomap/AssertionCuda.cuh>
#include <octomap/OcTreeKey.h>

namespace octomap {

  class KeyArrayCuda {
  public:
    
    CUDA_CALLABLE KeyArrayCuda (
      const int& maxSize = 100000) :
      maxSize(maxSize)
    {
      ray = new OcTreeKey[maxSize];
      reset();
    }

    CUDA_CALLABLE ~KeyArrayCuda () {
      delete ray;
    }
    
    CUDA_CALLABLE KeyArrayCuda(const KeyArrayCuda& other)
    {
      ray = other.ray;
      last = other.last;
      maxSize = other.maxSize;
    }
    
    __host__ void allocateDevice() {
      cudaCheckErrors(cudaMallocManaged(&ray, maxSize * sizeof(OcTreeKey)))
    }

    __host__ void freeDevice() {
      cudaCheckErrors(cudaFree(ray));
    }

    __host__ void copyToDevice(const KeyArrayCuda& other) {
      assert (maxSize == other.sizeMax());
      cudaCheckErrors(cudaMemcpy(ray, other.ray, maxSize * sizeof(OcTreeKey), cudaMemcpyHostToDevice));
      last = other.last;
    }

    __host__ void copyToHost(const KeyArrayCuda& other) {
      assert (maxSize == other.sizeMax());
      cudaCheckErrors(cudaMemcpy(ray, other.ray, maxSize * sizeof(OcTreeKey), cudaMemcpyDeviceToHost));
      last = other.last;
    }

    CUDA_CALLABLE KeyArrayCuda& operator=(const KeyArrayCuda& other){
      ray = other.ray;
      last = other.last;
      maxSize = other.maxSize;
      return *this;
    }
    
    CUDA_CALLABLE void reset() {
      last = 0;
    }
    
    CUDA_CALLABLE void addKey(const OcTreeKey& k) {
      assert(last != maxSize);
      ray[last] = k;
      ++last;
    }

    __device__ void addKeyAtomic(const OcTreeKey& k);

    CUDA_CALLABLE const OcTreeKey* begin() { return ray; }
    CUDA_CALLABLE const OcTreeKey* end() { return &ray[last]; }

    CUDA_CALLABLE int size() const { return last; }
    CUDA_CALLABLE int sizeMax() const { return maxSize; }
    
  private:
    OcTreeKey* ray;
    int last;
    int maxSize;
  };

  using KeyRayCuda = KeyArrayCuda;

  struct KeyValue {
    OcTreeKey key;
    void* value;
    KeyValue *next;
  };

  class KeyHashMapCuda {
  public:
    
    CUDA_CALLABLE KeyHashMapCuda () :
    {
      size = NPP_MAX_32U;
    }

    CUDA_CALLABLE ~KeyHashMapCuda () {
    }

    __host__ void allocateDevice() {
      table.count = entries;
      HANDLE_ERROR( cudaMalloc( (void**)&table.entries,
      entries * sizeof(Entry*)) );
      HANDLE_ERROR( cudaMemset( table.entries, 0,
      entries * sizeof(Entry*) ) );
      HANDLE_ERROR( cudaMalloc( (void**)&table.pool,
      elements * sizeof(Entry)) );
      cudaCheckErrors(cudaMallocManaged(&arr, arr_size);
    }

    __host__ void freeDevice() {
      cudaCheckErrors(cudaFree(arr));
    }

    __host__ void copyToDevice(const KeyHashMapCuda& other) {
      assert (maxRowSize == other.sizeMax());
      cudaCheckErrors(cudaMemcpy(arr, other.arr, arr_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    __host__ void copyToHost(const KeyHashMapCuda& other) {
      assert (maxRowSize == other.sizeMax());
      cudaCheckErrors(cudaMemcpy(arr, other.arr, arr_size * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    /*CUDA_CALLABLE void set(const int& val, const size_t& i, const size_t& j, const size_t& k) {
      arr[i + j * maxRowSize + k * maxRowSize] = val;
    }

    CUDA_CALLABLE int get(const size_t& i, const size_t& j, const size_t& k) {
      return arr[i][j][k];
    }*/

    //__device__ void setAtomic(const int& val, const size_t& i, const size_t& j, const size_t& k);

    //CUDA_CALLABLE const size_t* begin() { return arr; }
    //CUDA_CALLABLE const size_t* end() { return &arr[arr_size]; }
    
  private:
    KeyValue* map;
    KeyValue** entries;
    size_t size;
    bool host_alloc = {false};
  };
}
#endif
#endif