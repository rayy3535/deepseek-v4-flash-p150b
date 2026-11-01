// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC A3: the mHC map straight from the [1, M, n, H] residual, split over cores along the hidden width.
// For each block row m this core accumulates, over its hidden tiles t (FPU, fp32 DST):
//   DST l (l < n) += X_t @ W_{l,t}   X_t the residual tile (rows = lanes), W_{l,t} lane l's map weight tile;
//                                     row l of DST l is lane l's share of x @ fn, the other rows are discarded
//   DST n         += X_t @ X_t^T      its diagonal (l, l) is lane l's sum of squares (for the RMS norm)
// and packs the n + 1 partial tiles; the epilogue (kernels/hc_pre/, rows_from_map) sums the cores' partials, takes
// the diagonal / lane rows, applies 1 / rms (the norm has no weight, so it commutes with the map) and the bias.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

namespace {
constexpr uint32_t cb_streams = tt::CBIndex::c_0;
constexpr uint32_t cb_fn = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
}  // namespace

void kernel_main() {
    constexpr uint32_t lanes = get_compile_time_arg_val(0);
    const uint32_t nt = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_streams, cb_fn, cb_out);
    cb_wait_front(cb_fn, lanes * nt);
    for (uint32_t m = 0; m < num_rows; ++m) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < nt; ++i) {
            cb_wait_front(cb_streams, 1);
            reconfig_data_format(cb_fn, cb_streams);  // srcA <- in1 (weights, fp32), srcB <- in0 (residual)
            matmul_init(cb_streams, cb_fn);
            for (uint32_t l = 0; l < lanes; ++l) {
                matmul_tiles(cb_streams, cb_fn, 0, l * nt + i, l);
            }
            reconfig_data_format(cb_streams, cb_streams);
            matmul_init(cb_streams, cb_streams, 1);  // in1 transposed: X @ X^T
            matmul_tiles(cb_streams, cb_streams, 0, 0, lanes);
            cb_pop_front(cb_streams, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, lanes + 1);
        for (uint32_t d = 0; d <= lanes; ++d) {
            pack_tile(d, cb_out);
        }
        cb_push_back(cb_out, lanes + 1);
        tile_regs_release();
    }
    cb_pop_front(cb_fn, lanes * nt);
}
