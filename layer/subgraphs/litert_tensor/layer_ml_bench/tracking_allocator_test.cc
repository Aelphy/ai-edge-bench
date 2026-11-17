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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "include/xnnpack.h"

namespace layer_ml_bench {
namespace {

constexpr size_t kMiB = 1024 * 1024;

// Heap sampler reporting an empty heap, which leaves a MemoryTracker with
// only the growth recorded by its TrackingAllocator.
size_t EmptyHeap() { return 0; }

TEST(TrackingAllocatorTest, StartsWithNothingAllocated) {
  TrackingAllocator allocator;

  EXPECT_EQ(allocator.current_allocated(), 0);
  EXPECT_EQ(allocator.peak_allocated(), 0);
}

TEST(TrackingAllocatorTest, RecordAllocAccumulatesCurrentAndPeak) {
  TrackingAllocator allocator;

  allocator.RecordAlloc(100);
  allocator.RecordAlloc(200);

  EXPECT_EQ(allocator.current_allocated(), 300);
  EXPECT_EQ(allocator.peak_allocated(), 300);
}

TEST(TrackingAllocatorTest, RecordFreeLowersCurrentAndKeepsPeak) {
  TrackingAllocator allocator;
  allocator.RecordAlloc(300);

  allocator.RecordFree(50);

  EXPECT_EQ(allocator.current_allocated(), 250);
  EXPECT_EQ(allocator.peak_allocated(), 300);
}

TEST(TrackingAllocatorTest, RecordFreeBeyondCurrentClampsToZero) {
  TrackingAllocator allocator;
  allocator.RecordAlloc(300);

  allocator.RecordFree(500);

  EXPECT_EQ(allocator.current_allocated(), 0);
  EXPECT_EQ(allocator.peak_allocated(), 300);
}

TEST(TrackingAllocatorTest, ResetClearsCurrentAndPeak) {
  TrackingAllocator allocator;
  allocator.RecordAlloc(300);

  allocator.Reset();

  EXPECT_EQ(allocator.current_allocated(), 0);
  EXPECT_EQ(allocator.peak_allocated(), 0);
}

TEST(TrackingAllocatorTest, ResetPeakToCurrentDropsPeakToLiveBytes) {
  TrackingAllocator allocator;
  allocator.RecordAlloc(10240);
  allocator.RecordFree(8192);

  allocator.ResetPeakToCurrent();

  EXPECT_EQ(allocator.current_allocated(), 2048);
  EXPECT_EQ(allocator.peak_allocated(), 2048);
}

TEST(TrackingAllocatorTest, ResetPeakToCurrentKeepsTrackingLiveBlocks) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* live_block = xnn_alloc.allocate(xnn_alloc.context, 2048);
  ASSERT_NE(live_block, nullptr);

  allocator.ResetPeakToCurrent();
  xnn_alloc.deallocate(xnn_alloc.context, live_block);

  EXPECT_EQ(allocator.current_allocated(), 0);
}

TEST(TrackingAllocatorTest, AllocateTracksRequestedBytes) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();

  void* first_block = xnn_alloc.allocate(xnn_alloc.context, 1024);
  void* second_block = xnn_alloc.allocate(xnn_alloc.context, 2048);

  EXPECT_NE(first_block, nullptr);
  EXPECT_NE(second_block, nullptr);
  EXPECT_EQ(allocator.current_allocated(), 3072);
  EXPECT_EQ(allocator.peak_allocated(), 3072);
  xnn_alloc.deallocate(xnn_alloc.context, first_block);
  xnn_alloc.deallocate(xnn_alloc.context, second_block);
}

TEST(TrackingAllocatorTest, DeallocateReleasesBlockBytesAndKeepsPeak) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* first_block = xnn_alloc.allocate(xnn_alloc.context, 1024);
  void* second_block = xnn_alloc.allocate(xnn_alloc.context, 2048);

  xnn_alloc.deallocate(xnn_alloc.context, first_block);

  EXPECT_EQ(allocator.current_allocated(), 2048);
  EXPECT_EQ(allocator.peak_allocated(), 3072);
  xnn_alloc.deallocate(xnn_alloc.context, second_block);
}

TEST(TrackingAllocatorTest, DeallocateNullLeavesAccountingUnchanged) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* block = xnn_alloc.allocate(xnn_alloc.context, 64);

  xnn_alloc.deallocate(xnn_alloc.context, nullptr);
  xnn_alloc.aligned_deallocate(xnn_alloc.context, nullptr);

  EXPECT_EQ(allocator.current_allocated(), 64);
  xnn_alloc.deallocate(xnn_alloc.context, block);
}

TEST(TrackingAllocatorTest, AlignedAllocateHonorsAlignmentAndTracksBytes) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();

  // 64 bytes is the cache-line alignment requested for SIMD buffers.
  void* block = xnn_alloc.aligned_allocate(xnn_alloc.context, 64, 4096);

  ASSERT_NE(block, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(block) % 64, 0);
  EXPECT_EQ(allocator.current_allocated(), 4096);
  EXPECT_EQ(allocator.peak_allocated(), 4096);
  xnn_alloc.aligned_deallocate(xnn_alloc.context, block);
}

TEST(TrackingAllocatorTest, AlignedAllocateAcceptsAlignmentBelowPointerSize) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();

  void* block = xnn_alloc.aligned_allocate(xnn_alloc.context, 2, 128);

  ASSERT_NE(block, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(block) % sizeof(void*), 0);
  EXPECT_EQ(allocator.current_allocated(), 128);
  xnn_alloc.aligned_deallocate(xnn_alloc.context, block);
}

TEST(TrackingAllocatorTest, AlignedDeallocateReleasesBlockBytesAndKeepsPeak) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* block = xnn_alloc.aligned_allocate(xnn_alloc.context, 64, 4096);

  xnn_alloc.aligned_deallocate(xnn_alloc.context, block);

  EXPECT_EQ(allocator.current_allocated(), 0);
  EXPECT_EQ(allocator.peak_allocated(), 4096);
}

TEST(TrackingAllocatorTest, ReallocateNullAllocatesNewBlock) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();

  void* block = xnn_alloc.reallocate(xnn_alloc.context, nullptr, 1000);

  EXPECT_NE(block, nullptr);
  EXPECT_EQ(allocator.current_allocated(), 1000);
  xnn_alloc.deallocate(xnn_alloc.context, block);
}

TEST(TrackingAllocatorTest, ReallocateGrowTracksNewSizeAndKeepsContents) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* block = xnn_alloc.allocate(xnn_alloc.context, 1000);
  ASSERT_NE(block, nullptr);
  std::memset(block, 0xAB, 1000);

  void* grown_block = xnn_alloc.reallocate(xnn_alloc.context, block, 5000);

  ASSERT_NE(grown_block, nullptr);
  EXPECT_EQ(std::string(static_cast<const char*>(grown_block), 1000),
            std::string(1000, '\xAB'));
  EXPECT_EQ(allocator.current_allocated(), 5000);
  EXPECT_EQ(allocator.peak_allocated(), 5000);
  xnn_alloc.deallocate(xnn_alloc.context, grown_block);
}

TEST(TrackingAllocatorTest, ReallocateShrinkTracksNewSizeAndKeepsPeak) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* block = xnn_alloc.allocate(xnn_alloc.context, 5000);
  ASSERT_NE(block, nullptr);
  std::memset(block, 0xAB, 5000);

  void* shrunk_block = xnn_alloc.reallocate(xnn_alloc.context, block, 2000);

  ASSERT_NE(shrunk_block, nullptr);
  EXPECT_EQ(std::string(static_cast<const char*>(shrunk_block), 2000),
            std::string(2000, '\xAB'));
  EXPECT_EQ(allocator.current_allocated(), 2000);
  EXPECT_EQ(allocator.peak_allocated(), 5000);
  xnn_alloc.deallocate(xnn_alloc.context, shrunk_block);
}

TEST(TrackingAllocatorTest, ReallocateToZeroFreesBlock) {
  TrackingAllocator allocator;
  xnn_allocator xnn_alloc = allocator.AsXnnAllocator();
  void* block = xnn_alloc.allocate(xnn_alloc.context, 1000);

  void* freed_block = xnn_alloc.reallocate(xnn_alloc.context, block, 0);

  EXPECT_EQ(freed_block, nullptr);
  EXPECT_EQ(allocator.current_allocated(), 0);
}

TEST(TrackingAllocatorTest, IsNeitherCopyableNorMovable) {
  EXPECT_FALSE(std::is_copy_constructible_v<TrackingAllocator>);
  EXPECT_FALSE(std::is_copy_assignable_v<TrackingAllocator>);
  EXPECT_FALSE(std::is_move_constructible_v<TrackingAllocator>);
  EXPECT_FALSE(std::is_move_assignable_v<TrackingAllocator>);
}

TEST(GlobalTrackingAllocatorTest, TracksXnnpackAllocationsAfterInitialization) {
  // The repeated call checks that initialization is idempotent.
  EnsureXnnpackInitializedWithTracking();
  EnsureXnnpackInitializedWithTracking();
  TrackingAllocator& allocator = GetGlobalTrackingAllocator();
  const size_t bytes_before = allocator.current_allocated();

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(
      xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph),
      xnn_status_success);
  const size_t bytes_with_subgraph = allocator.current_allocated();
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);

  EXPECT_GT(bytes_with_subgraph, bytes_before);
  EXPECT_EQ(allocator.current_allocated(), bytes_before);
}

TEST(MemoryTrackerTest, ReportsZeroBeforeAnyWindow) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator, EmptyHeap);

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 0.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 0.0);
}

TEST(MemoryTrackerTest, ReportsTrackedGrowthWithinWindow) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator, EmptyHeap);

  tracker.Start();
  allocator.RecordAlloc(4 * kMiB);
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 4.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 4.0);
}

TEST(MemoryTrackerTest, PeakIncludesBytesFreedWithinWindow) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator, EmptyHeap);

  tracker.Start();
  allocator.RecordAlloc(8 * kMiB);
  allocator.RecordFree(6 * kMiB);
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 2.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 8.0);
}

TEST(MemoryTrackerTest, ExcludesBytesAllocatedBeforeStart) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator, EmptyHeap);
  allocator.RecordAlloc(8 * kMiB);

  tracker.Start();
  allocator.RecordAlloc(2 * kMiB);
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 2.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 2.0);
}

TEST(MemoryTrackerTest, ExcludesBytesAllocatedAfterStop) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator, EmptyHeap);

  tracker.Start();
  allocator.RecordAlloc(2 * kMiB);
  tracker.Stop();
  allocator.RecordAlloc(8 * kMiB);

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 2.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 2.0);
}

TEST(MemoryTrackerTest, PeakDoesNotLeakFromEarlierTrackerWindow) {
  TrackingAllocator allocator;
  MemoryTracker large_tracker(&allocator, EmptyHeap);
  large_tracker.Start();
  allocator.RecordAlloc(16 * kMiB);
  allocator.RecordFree(16 * kMiB);
  large_tracker.Stop();
  MemoryTracker small_tracker(&allocator, EmptyHeap);

  small_tracker.Start();
  allocator.RecordAlloc(1 * kMiB);
  allocator.RecordFree(1 * kMiB);
  small_tracker.Stop();

  EXPECT_DOUBLE_EQ(small_tracker.PeakAllocatedMB(), 1.0);
  EXPECT_DOUBLE_EQ(small_tracker.AllocatedDeltaMB(), 0.0);
}

TEST(MemoryTrackerTest, ReportsGrowthOfInjectedHeapSampler) {
  TrackingAllocator allocator;
  size_t heap_bytes = 10 * kMiB;
  MemoryTracker tracker(&allocator, [&heap_bytes]() { return heap_bytes; });

  tracker.Start();
  heap_bytes += 12 * kMiB;
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 12.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 12.0);
}

TEST(MemoryTrackerTest, ShrinkingHeapReportsZeroDelta) {
  TrackingAllocator allocator;
  size_t heap_bytes = 22 * kMiB;
  MemoryTracker tracker(&allocator, [&heap_bytes]() { return heap_bytes; });

  tracker.Start();
  heap_bytes -= 12 * kMiB;
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 0.0);
  EXPECT_DOUBLE_EQ(tracker.PeakAllocatedMB(), 0.0);
}

TEST(MemoryTrackerTest, DeltaIsTrackedGrowthWhenItExceedsHeapGrowth) {
  TrackingAllocator allocator;
  size_t heap_bytes = 10 * kMiB;
  MemoryTracker tracker(&allocator, [&heap_bytes]() { return heap_bytes; });

  tracker.Start();
  heap_bytes += 3 * kMiB;
  allocator.RecordAlloc(5 * kMiB);
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 5.0);
}

TEST(MemoryTrackerTest, DeltaIsHeapGrowthWhenItExceedsTrackedGrowth) {
  TrackingAllocator allocator;
  size_t heap_bytes = 10 * kMiB;
  MemoryTracker tracker(&allocator, [&heap_bytes]() { return heap_bytes; });

  tracker.Start();
  heap_bytes += 7 * kMiB;
  allocator.RecordAlloc(5 * kMiB);
  tracker.Stop();

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 7.0);
}

TEST(MemoryTrackerTest, DefaultHeapSamplerNeverReportsLessThanTrackedGrowth) {
  TrackingAllocator allocator;
  MemoryTracker tracker(&allocator);

  tracker.Start();
  allocator.RecordAlloc(4 * kMiB);
  tracker.Stop();

  // The process heap sampler can only raise the reported values above the
  // tracked growth, so these bounds hold for any allocator behavior.
  EXPECT_GE(tracker.AllocatedDeltaMB(), 4.0);
  EXPECT_GE(tracker.PeakAllocatedMB(), 4.0);
}

TEST(MemoryTrackerTest, NullAllocatorTracksGlobalAllocator) {
  TrackingAllocator& global_allocator = GetGlobalTrackingAllocator();
  MemoryTracker tracker(/*allocator=*/nullptr, EmptyHeap);

  tracker.Start();
  global_allocator.RecordAlloc(4 * kMiB);
  tracker.Stop();
  global_allocator.RecordFree(4 * kMiB);

  EXPECT_DOUBLE_EQ(tracker.AllocatedDeltaMB(), 4.0);
}

TEST(MemoryTrackerTest, IsNeitherCopyableNorMovable) {
  EXPECT_FALSE(std::is_copy_constructible_v<MemoryTracker>);
  EXPECT_FALSE(std::is_copy_assignable_v<MemoryTracker>);
  EXPECT_FALSE(std::is_move_constructible_v<MemoryTracker>);
  EXPECT_FALSE(std::is_move_assignable_v<MemoryTracker>);
}

}  // namespace
}  // namespace layer_ml_bench
