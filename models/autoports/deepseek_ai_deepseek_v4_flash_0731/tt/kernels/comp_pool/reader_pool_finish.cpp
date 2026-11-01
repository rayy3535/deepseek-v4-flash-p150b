// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// data movement in for output column tile c (pool_finish_compute.cpp): every column's
// sum-of-squares tile, the pooled rows this core needs (its own; all of them for the Hadamard history), their norm
// weight / cos / sin rows (bf16 -> fp32, other rows zero), the RoPE rotation tile and Hadamard column c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE = 1024;

FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t stage1_addr = get_arg_val<uint32_t>(0);
    const uint32_t norm_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t trans_addr = get_arg_val<uint32_t>(4);
    const uint32_t had_addr = get_arg_val<uint32_t>(5);
    const uint32_t c_out = get_arg_val<uint32_t>(6);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t rope_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t rotate = get_compile_time_arg_val(2);
    constexpr auto st_args = TensorAccessorArgs<3>();
    constexpr auto norm_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<norm_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto had_args = TensorAccessorArgs<trans_args.next_compile_time_args_offset()>();
    const auto st = TensorAccessor(st_args, stage1_addr);
    const auto norm = TensorAccessor(norm_args, norm_addr);
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    const auto trans = TensorAccessor(trans_args, trans_addr);
    const auto had = TensorAccessor(had_args, had_addr);

    Noc noc;
    CircularBuffer sq_cb(0), pooled_cb(1), norm_cb(2), cos_cb(3), sin_cb(4), trans_cb(5), had_cb(6), stage_cb(7);
    constexpr uint32_t f32 = 4096, bf16 = 2048;
    constexpr uint32_t first = rotate ? 0 : 0;
    const uint32_t lo = rotate ? 0 : c_out, hi = rotate ? Wt : c_out + 1;  // pooled columns this core needs

    sq_cb.reserve_back(Wt);
    for (uint32_t t = 0; t < Wt; ++t) {
        noc.async_read(st, CoreLocalMem<uint32_t>(sq_cb.get_write_ptr() + t * f32), f32, {.page_id = Wt + t}, {});
    }
    pooled_cb.reserve_back(hi - lo);
    for (uint32_t t = lo; t < hi; ++t) {
        noc.async_read(st, CoreLocalMem<uint32_t>(pooled_cb.get_write_ptr() + (t - lo) * f32), f32, {.page_id = t}, {});
    }
    trans_cb.reserve_back(1);
    noc.async_read(trans, CoreLocalMem<uint32_t>(trans_cb.get_write_ptr()), f32, {.page_id = 0}, {});
    if constexpr (rotate) {
        had_cb.reserve_back(Wt);
        for (uint32_t k = 0; k < Wt; ++k) {
            noc.async_read(
                had, CoreLocalMem<uint32_t>(had_cb.get_write_ptr() + k * f32), f32, {.page_id = k * Wt + c_out}, {});
        }
    }

    // norm / cos / sin rows for columns lo .. hi - 1 (cos / sin indexed by the column's place in the RoPE tail)
    stage_cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* stage = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage_cb.get_write_ptr());
    auto row0 = [&](const auto& src, uint32_t page_id, volatile tt_l1_ptr uint32_t* dst) {
        for (uint32_t w = 0; w < TILE; ++w) {
            dst[w] = 0;
        }
        noc.async_read(src, CoreLocalMem<uint32_t>(stage_cb.get_write_ptr()), bf16, {.page_id = page_id}, {});
        noc.async_read_barrier();
        for (uint32_t col = 0; col < 32; ++col) {
            dst[at(0, col)] = static_cast<uint32_t>(stage[at(0, col)]) << 16;
        }
    };
    norm_cb.reserve_back(hi - lo);
    cos_cb.reserve_back(hi - lo);
    sin_cb.reserve_back(hi - lo);
    for (uint32_t t = lo; t < hi; ++t) {
        auto* nw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(norm_cb.get_write_ptr() + (t - lo) * f32);
        row0(norm, t, nw);
        auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cos_cb.get_write_ptr() + (t - lo) * f32);
        auto* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sin_cb.get_write_ptr() + (t - lo) * f32);
        if (t + rope_tiles >= Wt) {
            row0(cos, t + rope_tiles - Wt, cw);
            row0(sin, t + rope_tiles - Wt, sw);
        }
    }
    noc.async_read_barrier();
    sq_cb.push_back(Wt);
    pooled_cb.push_back(hi - lo);
    norm_cb.push_back(hi - lo);
    cos_cb.push_back(hi - lo);
    sin_cb.push_back(hi - lo);
    trans_cb.push_back(1);
    if constexpr (rotate) {
        had_cb.push_back(Wt);
    }
    (void)first;
}
