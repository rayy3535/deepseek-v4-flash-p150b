// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                    Byte-based data transfer APIs
// ======================================================================================
//
// Outside the Runtime Tensor graduation surface (HostTensor <-> MeshTensor).
// Explicitly provided for unit mesh (single device) ttnn::Tensor H<->D access.
// Exposed via ttnn/async_runtime.hpp (write_buffer / read_buffer) and have
// external dependencies.

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

// Writes every shard of a (possibly multi-device) host tensor into the same byte range of the matching
// device shard: [offset, offset + host shard size). The offset must be page aligned and the range must lie
// inside the device buffer. Fast dispatch only. Lets a host-paged weight pool fill one slot of a larger
// device tensor without rewriting the rest of it. pin: pin each host shard (cached) so the device reads it
// directly from host memory (IOMMU in DMA mode; otherwise the regular copy path).
void enqueue_write_tensor_region(
    distributed::MeshCommandQueue& queue,
    const HostTensor& host_tensor,
    const MeshTensor& device_tensor,
    std::size_t offset,
    bool blocking = false,
    bool pin = false);

// Several region writes into one device tensor in a single queue submission: host_tensors[i] goes to
// [offsets[i], offsets[i] + shard size) of every device shard (same rules as enqueue_write_tensor_region).
void enqueue_write_tensor_regions(
    distributed::MeshCommandQueue& queue,
    const std::vector<HostTensor>& host_tensors,
    const MeshTensor& device_tensor,
    const std::vector<std::size_t>& offsets,
    bool blocking = false,
    bool pin = false);

}  // namespace tt::tt_metal
