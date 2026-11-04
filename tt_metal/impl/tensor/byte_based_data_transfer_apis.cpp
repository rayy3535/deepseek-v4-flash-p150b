// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/byte_based_tensor_transfers.hpp>

#include "mesh_tensor_impl.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/pinned_memory.hpp>
#include "tt_metal/distributed/pinned_memory_cache.hpp"

namespace tt::tt_metal {

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_read_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(dst)
            .region(region)};
    queue.enqueue_read_shards(shard_data_transfers, device_tensor.impl().raw_mesh_buffer(), blocking);
}

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_write_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(const_cast<std::byte*>(src))
            .region(region)};
    queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), shard_data_transfers, false);
}

namespace {

void append_region_transfers(
    std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    distributed::MeshDevice& mesh_device,
    const HostTensor& host_tensor,
    std::size_t offset,
    bool pin) {
    const auto& host_buffer = host_tensor.buffer();
    for (const auto& coord : host_buffer.shard_coords()) {
        auto shard = host_buffer.get_shard(coord);
        if (!shard.has_value()) {
            continue;  // shard owned by another host
        }
        auto bytes = shard->view_bytes();
        auto transfer = distributed::ShardDataTransfer{distributed::MeshCoordinate(coord)}
                            .host_data(bytes.data())
                            .region(BufferRegion(offset, bytes.size()));
        if (pin) {
            // Pinned once per host buffer (cached); the device then reads the shard straight from host memory.
            // Needs the IOMMU in DMA mode; without it try_pin returns nullptr and the copy path is used.
            HostBuffer pinned_buf(*shard);
            auto range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(
                distributed::MeshCoordinate(coord), distributed::MeshCoordinate(coord)));
            if (auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
                    mesh_device,
                    range,
                    pinned_buf,
                    /*map_to_noc=*/true,
                    experimental::PinnedMemoryDeviceAccess::ReadOnly)) {
                experimental::ShardDataTransferSetPinnedMemory(transfer, std::move(pinned));
            }
        }
        shard_data_transfers.push_back(std::move(transfer));
    }
}

}  // namespace

void enqueue_write_tensor_regions(
    distributed::MeshCommandQueue& queue,
    const std::vector<HostTensor>& host_tensors,
    const MeshTensor& device_tensor,
    const std::vector<std::size_t>& offsets,
    bool blocking,
    bool pin) {
    TT_FATAL(
        MetalContext::instance().rtoptions().get_fast_dispatch(),
        "enqueue_write_tensor_regions requires fast dispatch");
    TT_FATAL(host_tensors.size() == offsets.size(), "one offset per host tensor");
    auto* mesh_device = device_tensor.impl().raw_mesh_buffer()->device();
    std::vector<distributed::ShardDataTransfer> shard_data_transfers;
    for (std::size_t i = 0; i < host_tensors.size(); ++i) {
        append_region_transfers(shard_data_transfers, *mesh_device, host_tensors[i], offsets[i], pin);
    }
    queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), shard_data_transfers, blocking);
}

void enqueue_write_tensor_region(
    distributed::MeshCommandQueue& queue,
    const HostTensor& host_tensor,
    const MeshTensor& device_tensor,
    std::size_t offset,
    bool blocking,
    bool pin) {
    // Fast dispatch takes the region's own bytes as the source; slow dispatch would index src by the offset.
    TT_FATAL(
        MetalContext::instance().rtoptions().get_fast_dispatch(), "enqueue_write_tensor_region requires fast dispatch");
    const auto& host_buffer = host_tensor.buffer();
    auto* mesh_device = device_tensor.impl().raw_mesh_buffer()->device();
    std::vector<distributed::ShardDataTransfer> shard_data_transfers;
    shard_data_transfers.reserve(host_buffer.shard_coords().size());
    for (const auto& coord : host_buffer.shard_coords()) {
        auto shard = host_buffer.get_shard(coord);
        if (!shard.has_value()) {
            continue;  // shard owned by another host
        }
        auto bytes = shard->view_bytes();
        auto transfer = distributed::ShardDataTransfer{distributed::MeshCoordinate(coord)}
                            .host_data(bytes.data())
                            .region(BufferRegion(offset, bytes.size()));
        if (pin) {
            // Pinned once per host buffer (cached); the device then reads the shard straight from host memory.
            // Needs the IOMMU in DMA mode; without it try_pin returns nullptr and the copy path is used.
            HostBuffer pinned_buf(*shard);
            auto range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(
                distributed::MeshCoordinate(coord), distributed::MeshCoordinate(coord)));
            if (auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
                    *mesh_device,
                    range,
                    pinned_buf,
                    /*map_to_noc=*/true,
                    experimental::PinnedMemoryDeviceAccess::ReadOnly)) {
                experimental::ShardDataTransferSetPinnedMemory(transfer, std::move(pinned));
            }
        }
        shard_data_transfers.push_back(std::move(transfer));
    }
    queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), shard_data_transfers, blocking);
}

}  // namespace tt::tt_metal
