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

#ifndef TENSOR_BENCHMARKS_LAYER_TRACKING_ALLOCATOR_H_
#define TENSOR_BENCHMARKS_LAYER_TRACKING_ALLOCATOR_H_

#include <cstddef>
#include <functional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "include/xnnpack.h"

namespace layer_ml_bench {

// Thread-safe memory allocation tracking wrapper conforming to xnn_allocator.
// Intercepts allocations, reallocations, and deallocations to maintain live
// current and peak byte counters for kernel initialization and runtime
// telemetry.
class TrackingAllocator {
 public:
  TrackingAllocator() = default;
  ~TrackingAllocator() = default;

  // Not copyable or movable: xnn_allocator callbacks capture `this`.
  TrackingAllocator(const TrackingAllocator&) = delete;
  TrackingAllocator& operator=(const TrackingAllocator&) = delete;

  // Manual accounting hooks (e.g. for non-pointer telemetry or testing).
  void RecordAlloc(size_t size);
  void RecordFree(size_t size);
  void Reset();

  // Resets the high-water mark to current_allocated() without clearing active
  // pointer tracking, scoping peak telemetry to a single measurement window.
  void ResetPeakToCurrent();

  // Returns currently outstanding allocated bytes.
  size_t current_allocated() const;

  // Returns peak allocated bytes recorded since construction or Reset().
  size_t peak_allocated() const;

  // Returns an xnn_allocator struct configured with tracking callbacks bound
  // to this TrackingAllocator instance.
  xnn_allocator AsXnnAllocator();

 private:
  void TrackAlloc(void* ptr, size_t size);
  void TrackFree(void* ptr);

  static void* XnnAllocate(void* context, size_t size);
  static void* XnnReallocate(void* context, void* pointer, size_t size);
  static void XnnDeallocate(void* context, void* pointer);
  static void* XnnAlignedAllocate(void* context, size_t alignment, size_t size);
  static void XnnAlignedDeallocate(void* context, void* pointer);

  mutable absl::Mutex mutex_;
  size_t current_allocated_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t peak_allocated_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::flat_hash_map<void*, size_t> allocations_ ABSL_GUARDED_BY(mutex_);
};

// Returns the global process-wide singleton TrackingAllocator.
TrackingAllocator& GetGlobalTrackingAllocator();

// Thread-safe one-time initialization ensuring xnn_initialize() is called with
// the global TrackingAllocator before any backend or un-instrumented code can
// call xnn_initialize(nullptr).
void EnsureXnnpackInitializedWithTracking();

// Dual-mode memory telemetry tracker sampling both TrackingAllocator and
// TCMalloc (or /proc/self/statm fallback).
//
// Lifecycle:
// Peak allocations (packed weights, workspace, execution buffers) occur during
// backend Setup() and the first warm-up iteration. Steady-state execution
// performs zero dynamic heap allocations. Therefore, Start() is called
// before backend Setup(), and Stop() is called after Setup() and one warm-up
// execution.
class MemoryTracker {
 public:
  // Returns the bytes currently allocated on the process heap.
  using HeapSampler = std::function<size_t()>;

  // A null `allocator` tracks GetGlobalTrackingAllocator(). A null
  // `heap_sampler` samples the process heap through the allocator's own
  // statistics where available and the resident set size otherwise.
  // `allocator` must outlive the tracker.
  explicit MemoryTracker(TrackingAllocator* allocator = nullptr,
                         HeapSampler heap_sampler = nullptr);

  // Not copyable or movable: a tracker owns one Start()/Stop() measurement
  // window.
  MemoryTracker(const MemoryTracker&) = delete;
  MemoryTracker& operator=(const MemoryTracker&) = delete;

  // Call before backend Setup() and runtime initialization.
  void Start();

  // Call after backend Setup() and one warm-up execution.
  void Stop();

  // Peak memory allocated across layer initialization, weight packing, and
  // warmup (MB).
  double PeakAllocatedMB() const;

  // Net heap delta allocated across setup and execution (MB).
  double AllocatedDeltaMB() const;

 private:
  TrackingAllocator* allocator_;
  HeapSampler heap_sampler_;
  size_t start_heap_bytes_ = 0;
  size_t stop_heap_bytes_ = 0;
  size_t start_xnn_bytes_ = 0;
  size_t stop_xnn_bytes_ = 0;
  size_t peak_xnn_bytes_ = 0;
};

}  // namespace layer_ml_bench

#endif  // TENSOR_BENCHMARKS_LAYER_TRACKING_ALLOCATOR_H_
