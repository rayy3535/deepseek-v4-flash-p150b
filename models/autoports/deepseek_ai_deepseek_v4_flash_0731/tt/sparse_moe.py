"""Single-chip active-expert MoE for DeepSeek-V4-Flash-0731.

The generic multi-chip DeepSeek MoE packs every expert into its dispatch-specific
layout before it can execute. That is useful on an expert-parallel mesh, but is
unnecessary and prohibitively expensive for this functional 1x1 stage. Here the
stock ``ttnn.sparse_matmul`` path consumes the router's dense top-6 mask directly,
so inactive token/expert pairs do no matmul work. The shared expert is a dense
SwiGLU branch added after the routed reduction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn


@dataclass(frozen=True)
class SparseMoEWeights:
    gate: ttnn.Tensor
    up: ttnn.Tensor
    down: ttnn.Tensor
    shared_gate: ttnn.Tensor
    shared_up: ttnn.Tensor
    shared_down: ttnn.Tensor


def _program_config(m: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Use an 8x8-safe 1-D multicast geometry for the p300 worker grid."""

    n_tiles = math.ceil(n / 32)
    best_x = best_y = best_cores = 1
    for cores in range(1, min(64, n_tiles) + 1):
        if n_tiles % cores:
            continue
        for y in range(1, 9):
            if cores % y == 0 and cores // y <= 8 and cores > best_cores:
                best_x, best_y, best_cores = cores // y, y, cores
                break
    per_core_n = n_tiles // best_cores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_x, best_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class DeepSeekSparseMoE:
    """Trace-safe top-k routed experts plus one shared expert."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        expert_w1: torch.Tensor,
        expert_w2: torch.Tensor,
        expert_w3: torch.Tensor,
        shared_w1: torch.Tensor,
        shared_w2: torch.Tensor,
        shared_w3: torch.Tensor,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        limit: float,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> None:
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.limit = limit

        expected_gate = (num_experts, intermediate_size, hidden_size)
        expected_down = (num_experts, hidden_size, intermediate_size)
        if tuple(expert_w1.shape) != expected_gate or tuple(expert_w3.shape) != expected_gate:
            raise ValueError(
                f"routed gate/up shapes must be {expected_gate}, got "
                f"{tuple(expert_w1.shape)} and {tuple(expert_w3.shape)}"
            )
        if tuple(expert_w2.shape) != expected_down:
            raise ValueError(f"routed down shape must be {expected_down}, got {tuple(expert_w2.shape)}")

        mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        def upload(name: str, weight: torch.Tensor) -> ttnn.Tensor:
            start = time.perf_counter()
            logger.info(f"Uploading DeepSeek sparse-MoE {name}: shape={tuple(weight.shape)} dtype={weights_dtype}")
            # BF16's generic strided conversion is prohibitively slow for a
            # 4 GiB transpose view. Stage one projection at a time so the bulk
            # upload remains fast without retaining three additional copies.
            staged = weight.contiguous() if weights_dtype == ttnn.bfloat16 else weight
            result = ttnn.from_torch(
                staged,
                device=mesh_device,
                mesh_mapper=mapper,
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logger.info(f"Uploaded DeepSeek sparse-MoE {name} in {time.perf_counter() - start:.2f}s")
            return result

        self.weights = SparseMoEWeights(
            gate=upload("routed gate", expert_w1.transpose(-2, -1).unsqueeze(0)),
            up=upload("routed up", expert_w3.transpose(-2, -1).unsqueeze(0)),
            down=upload("routed down", expert_w2.transpose(-2, -1).unsqueeze(0)),
            shared_gate=upload("shared gate", shared_w1.transpose(-2, -1)),
            shared_up=upload("shared up", shared_w3.transpose(-2, -1)),
            shared_down=upload("shared down", shared_w2.transpose(-2, -1)),
        )
        self.gate_up_program = _program_config(32, intermediate_size)
        self.down_program = _program_config(32, hidden_size)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Indexed sparse_matmul does not read sparsity values, but retains the
        # dense E-wide operand as an explicit shape contract.
        self.sparsity = ttnn.ones(
            [1, 1, 1, num_experts],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _swiglu(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        return ttnn.mul(ttnn.silu(gate), up)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run up to 32 tokens with compact device-resident top-k metadata."""

        seq = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size) or not 1 <= seq <= 32:
            raise ValueError(f"MoE input must be [1,1,S,{self.hidden_size}] with 1 <= S <= 32")
        expected_routing = (1, 1, seq, self.top_k)
        if tuple(selected_experts.shape) != expected_routing or tuple(routing_weights.shape) != expected_routing:
            raise ValueError(
                f"selected experts and weights must both be {expected_routing}, got "
                f"{tuple(selected_experts.shape)} and {tuple(routing_weights.shape)}"
            )
        common = {
            "sparsity": self.sparsity,
            "is_input_b_sparse": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_tile": ttnn.Tile([32, 32]),
            "dtype": ttnn.bfloat16,
            "compute_kernel_config": self.compute_kernel_config,
        }
        routed_tokens = []
        for token_idx in range(seq):
            token = ttnn.slice(
                hidden_states,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.hidden_size],
            )
            indices = ttnn.slice(
                selected_experts,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.top_k],
            )
            indices = ttnn.to_layout(ttnn.reshape(indices, [1, 1, 1, self.top_k]), ttnn.ROW_MAJOR_LAYOUT)
            gate = ttnn.sparse_matmul(
                token,
                self.weights.gate,
                indices=indices,
                program_config=self.gate_up_program,
                **common,
            )
            up = ttnn.sparse_matmul(
                token,
                self.weights.up,
                indices=indices,
                program_config=self.gate_up_program,
                **common,
            )
            gate = ttnn.reshape(gate, (self.top_k, 1, self.intermediate_size))
            up = ttnn.reshape(up, (self.top_k, 1, self.intermediate_size))
            activated = ttnn.reshape(
                self._swiglu(gate, up),
                (1, self.top_k, 1, self.intermediate_size),
            )
            routed = ttnn.sparse_matmul(
                activated,
                self.weights.down,
                indices=indices,
                program_config=self.down_program,
                is_input_a_sparse=True,
                **common,
            )
            routed = ttnn.reshape(routed, (self.top_k, 1, self.hidden_size))
            token_weights = ttnn.slice(
                routing_weights,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.top_k],
            )
            routed = ttnn.mul(routed, ttnn.reshape(token_weights, (self.top_k, 1, 1)))
            routed_tokens.append(ttnn.reshape(ttnn.sum(routed, dim=0), (1, 1, 1, self.hidden_size)))
        routed = routed_tokens[0] if len(routed_tokens) == 1 else ttnn.concat(routed_tokens, dim=2)

        shared_gate = ttnn.linear(
            hidden_states,
            self.weights.shared_gate,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared_up = ttnn.linear(
            hidden_states,
            self.weights.shared_up,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared = self._swiglu(shared_gate, shared_up)
        shared = ttnn.linear(
            shared,
            self.weights.shared_down,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(routed, shared)
