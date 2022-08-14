#pragma once

#include <vector>

#include "Common.cuh"

template <class T>
class MemoryBuffer
{
   public:
    MemoryBuffer(size_t size);
    MemoryBuffer(size_t size, const T clone);
    MemoryBuffer(std::vector<T>& data);
    ~MemoryBuffer();

    size_t size() const { return mSize; }
    T* data() { return mCPUBuffer.data(); }
    T* gpu() { return mGPUBuffer; }
    void toDevice();
    void toHost();
    void clear();
    void fill(char c);

    T& operator[](int i) { return mCPUBuffer[i]; }

    operator T*() { return mGPUBuffer; }

   private:
    size_t mSize;
    std::vector<T> mCPUBuffer;
    T* mGPUBuffer;
};

template <class T>
inline MemoryBuffer<T>::MemoryBuffer(size_t size)
    : mSize(size), mCPUBuffer(size)
{
    memset(mCPUBuffer.data(), 0, sizeof(T) * mSize);
    check(cudaMalloc((void**)&mGPUBuffer, sizeof(T) * size));
}

template <class T>
inline MemoryBuffer<T>::MemoryBuffer(size_t size, const T clone)
    : mSize(size), mCPUBuffer(size, clone)
{
    check(cudaMalloc((void**)&mGPUBuffer, sizeof(T) * size));
}

template <class T>
inline MemoryBuffer<T>::MemoryBuffer(std::vector<T>& data)
    : mSize(data.size()), mCPUBuffer(data)
{
    check(cudaMalloc((void**)&mGPUBuffer, sizeof(T) * mSize));
}

template <class T>
inline MemoryBuffer<T>::~MemoryBuffer()
{
    check(cudaFree((void*)mGPUBuffer));
}

template <class T>
inline void MemoryBuffer<T>::toDevice()
{
    check(cudaMemcpy(mGPUBuffer, mCPUBuffer.data(), sizeof(T) * mSize,
                     cudaMemcpyHostToDevice));
}

template <class T>
inline void MemoryBuffer<T>::toHost()
{
    check(cudaMemcpy(mCPUBuffer.data(), mGPUBuffer, sizeof(T) * mSize,
                     cudaMemcpyDeviceToHost));
}

template <class T>
inline void MemoryBuffer<T>::clear()
{
    memset(mCPUBuffer.data(), 0, mCPUBuffer.size() * sizeof(T));
}

template <class T>
inline void MemoryBuffer<T>::fill(char c)
{
    memset(mCPUBuffer.data(), c, mCPUBuffer.size() * sizeof(T));
}