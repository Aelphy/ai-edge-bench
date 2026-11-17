// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensor/benchmarks/layer/tracking_allocator.h"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"
#include "include/xnnpack.h"

namespace layer_ml_bench {
namespace {

size_t SampleHeapAllocatedBytes() {
#if defined(__linux__)
  std::FILE* fp = std::fopen("/proc/self/statm", "rt");
  if (fp != nullptr) {
    size_t vsize = 0;
    size_t rss = 0;
    int scanned = std::fscanf(fp, "%zu %zu", &vsize, &rss);
    std::fclose(fp);
    if (scanned >= 2) {
      return rss * static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
  }
#endif
  return 0;
}

}  // namespace

void TrackingAllocator::RecordAlloc(size_t size) {
  absl::MutexLock lock(&mutex_);
  current_allocated_ += size;
  if (current_allocated_ > peak_allocated_) {
    peak_allocated_ = current_allocated_;
  }
}

void TrackingAllocator::RecordFree(size_t size) {
  absl::MutexLock lock(&mutex_);
  if (size >= current_allocated_) {
    current_allocated_ = 0;
  } else {
    current_allocated_ -= size;
  }
}

void TrackingAllocator::Reset() {
  absl::MutexLock lock(&mutex_);
  current_allocated_ = 0;
  peak_allocated_ = 0;
  allocations_.clear();
}

void TrackingAllocator::ResetPeakToCurrent() {
  absl::MutexLock lock(&mutex_);
  peak_allocated_ = current_allocated_;
}

size_t TrackingAllocator::current_allocated() const {
  absl::MutexLock lock(&mutex_);
  return current_allocated_;
}

size_t TrackingAllocator::peak_allocated() const {
  absl::MutexLock lock(&mutex_);
  return peak_allocated_;
}

void TrackingAllocator::TrackAlloc(void* ptr, size_t size) {
  absl::MutexLock lock(&mutex_);
  allocations_[ptr] = size;
  current_allocated_ += size;
  if (current_allocated_ > peak_allocated_) {
    peak_allocated_ = current_allocated_;
  }
}

void TrackingAllocator::TrackFree(void* ptr) {
  absl::MutexLock lock(&mutex_);
  auto it = allocations_.find(ptr);
  if (it != allocations_.end()) {
    size_t size = it->second;
    allocations_.erase(it);
    if (size >= current_allocated_) {
      current_allocated_ = 0;
    } else {
      current_allocated_ -= size;
    }
  }
}

void* TrackingAllocator::XnnAllocate(void* context, size_t size) {
  auto* self = static_cast<TrackingAllocator*>(context);
  void* ptr = std::malloc(size);
  if (ptr != nullptr && self != nullptr) {
    self->TrackAlloc(ptr, size);
  }
  return ptr;
}

void* TrackingAllocator::XnnAlignedAllocate(void* context, size_t alignment,
                                            size_t size) {
  auto* self = static_cast<TrackingAllocator*>(context);
  void* ptr = nullptr;
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  if (ptr != nullptr && self != nullptr) {
    self->TrackAlloc(ptr, size);
  }
  return ptr;
}

void* TrackingAllocator::XnnReallocate(void* context, void* pointer,
                                       size_t size) {
  auto* self = static_cast<TrackingAllocator*>(context);
  if (pointer == nullptr) {
    return XnnAllocate(context, size);
  }
  if (size == 0) {
    XnnDeallocate(context, pointer);
    return nullptr;
  }

  size_t old_size = 0;
  if (self != nullptr) {
    absl::MutexLock lock(&self->mutex_);
    auto it = self->allocations_.find(pointer);
    if (it != self->allocations_.end()) {
      old_size = it->second;
      self->allocations_.erase(it);
    }
  }

  void* new_ptr = std::realloc(pointer, size);
  if (new_ptr == nullptr) {
    if (self != nullptr && old_size > 0) {
      absl::MutexLock lock(&self->mutex_);
      self->allocations_[pointer] = old_size;
    }
    return nullptr;
  }

  if (self != nullptr) {
    absl::MutexLock lock(&self->mutex_);
    self->allocations_[new_ptr] = size;
    if (size >= old_size) {
      self->current_allocated_ += (size - old_size);
      if (self->current_allocated_ > self->peak_allocated_) {
        self->peak_allocated_ = self->current_allocated_;
      }
    } else {
      const size_t freed = old_size - size;
      self->current_allocated_ = (freed >= self->current_allocated_)
                                     ? 0
                                     : (self->current_allocated_ - freed);
    }
  }
  return new_ptr;
}

void TrackingAllocator::XnnDeallocate(void* context, void* pointer) {
  if (pointer == nullptr) return;
  auto* self = static_cast<TrackingAllocator*>(context);
  if (self != nullptr) {
    self->TrackFree(pointer);
  }
  std::free(pointer);
}

void TrackingAllocator::XnnAlignedDeallocate(void* context, void* pointer) {
  if (pointer == nullptr) return;
  auto* self = static_cast<TrackingAllocator*>(context);
  if (self != nullptr) {
    self->TrackFree(pointer);
  }
  std::free(pointer);
}

xnn_allocator TrackingAllocator::AsXnnAllocator() {
  xnn_allocator alloc = {};
  alloc.context = this;
  alloc.allocate = &TrackingAllocator::XnnAllocate;
  alloc.reallocate = &TrackingAllocator::XnnReallocate;
  alloc.deallocate = &TrackingAllocator::XnnDeallocate;
  alloc.aligned_allocate = &TrackingAllocator::XnnAlignedAllocate;
  alloc.aligned_deallocate = &TrackingAllocator::XnnAlignedDeallocate;
  return alloc;
}

TrackingAllocator& GetGlobalTrackingAllocator() {
  static TrackingAllocator* global_allocator = new TrackingAllocator();
  return *global_allocator;
}

void EnsureXnnpackInitializedWithTracking() {
  static absl::once_flag init_flag;
  absl::call_once(init_flag, []() {
    xnn_allocator alloc = GetGlobalTrackingAllocator().AsXnnAllocator();
    static_cast<void>(xnn_initialize(&alloc));
  });
}

MemoryTracker::MemoryTracker(TrackingAllocator* allocator,
                             HeapSampler heap_sampler)
    : allocator_(allocator != nullptr ? allocator
                                      : &GetGlobalTrackingAllocator()),
      heap_sampler_(std::move(heap_sampler)) {}

void MemoryTracker::Start() {
  start_heap_bytes_ =
      heap_sampler_ ? heap_sampler_() : SampleHeapAllocatedBytes();
  allocator_->ResetPeakToCurrent();
  start_xnn_bytes_ = allocator_->current_allocated();
  peak_xnn_bytes_ = start_xnn_bytes_;
}

void MemoryTracker::Stop() {
  stop_heap_bytes_ =
      heap_sampler_ ? heap_sampler_() : SampleHeapAllocatedBytes();
  stop_xnn_bytes_ = allocator_->current_allocated();
  peak_xnn_bytes_ = allocator_->peak_allocated();
}

double MemoryTracker::PeakAllocatedMB() const {
  double xnn_peak_delta_mb = 0.0;
  if (peak_xnn_bytes_ > start_xnn_bytes_) {
    xnn_peak_delta_mb =
        static_cast<double>(peak_xnn_bytes_ - start_xnn_bytes_) /
        (1024.0 * 1024.0);
  }
  return std::max(xnn_peak_delta_mb, AllocatedDeltaMB());
}

double MemoryTracker::AllocatedDeltaMB() const {
  double heap_delta_mb = 0.0;
  if (stop_heap_bytes_ > start_heap_bytes_) {
    heap_delta_mb = static_cast<double>(stop_heap_bytes_ - start_heap_bytes_) /
                    (1024.0 * 1024.0);
  }
  double xnn_delta_mb = 0.0;
  if (stop_xnn_bytes_ > start_xnn_bytes_) {
    xnn_delta_mb = static_cast<double>(stop_xnn_bytes_ - start_xnn_bytes_) /
                   (1024.0 * 1024.0);
  }
  return std::max(heap_delta_mb, xnn_delta_mb);
}

}  // namespace layer_ml_bench
