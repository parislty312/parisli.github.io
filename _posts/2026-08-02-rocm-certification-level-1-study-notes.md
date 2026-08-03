---
layout: post
title: "ROCm™ Certification – Level 1: Complete Study Notes"
subtitle: "An introduction to AMD GPUs and ROCm for AI and HPC, covering GPU architecture, ROCm libraries, HIP programming, and performance optimization."
date: 2026-08-02 09:00:00 -0700
categories: [AI Systems]
tags: [ROCm, AMD GPUs, HIP, Performance Optimization]
permalink: /technical-blog/rocm-certification-level-1-study-notes.html
summary: "Complete study notes from AMD Advancing AI 2026 on MI300X architecture, the ROCm software stack, HIP programming, and practical GPU performance optimization."
---

> Compiled from the in-person certification session at AMD Advancing AI 2026, San Francisco

---

# Module 2 — AMD GPU Architecture and Execution Model

## 1. The GPU Overview: Family → Package → CU

From the Instinct lineage down to the engines and memory inside a single Compute Unit (CU):

- AMD Instinct™ lineage (MI100 → MI400: each generation adds compute, memory, and new precisions — now on an annual cadence)
- MI300 anatomy — package to Compute Unit
- The SIMD unit — the CU's vector engine
- The MFMA matrix engine
- Memory hierarchy — registers to HBM

## 2. MI300X Anatomy — Package / XCD / Compute Unit

Three levels, each panel zooming into the one before it:

**① Full Package**
- Package = 8 XCDs + 8 HBM3 stacks + PCIe 5 + Infinity Fabric + Infinity Cache
- 192 GB HBM3 (8 stacks)
- 256 MB Infinity Cache — on the I/O dies

**② One XCD (Accelerator Complex Die)**
- XCD = 38 CUs + L2
- 38 active CUs per XCD (304 active CUs total)
- 4 MB L2 cache — shared by the die's CUs

**③ One CU (Compute Unit)**
- CU = 4 SIMDs × 16 ALUs + MFMA + LDS + L1
- 64 ALUs + MFMA — the building block of GPU compute
- LDS (shared memory) 64 KB; L1 cache 32 KB; scalar unit; register file

## 3. SIMD Unit — The CU's Vector Engine

- Each SIMD unit is 16 lanes wide; 1 lane = 1 ALU (the arithmetic logic unit that does the math)
- The 16 lanes run the same instruction on 16 different data elements
- A Compute Unit packs 4 SIMD units: ONE CU = 4 SIMDs × 16 lanes = 64 ALUs
- One instruction issued per SIMD

## 4. MFMA — One Instruction, One Full Matrix Tile

- **What it is**: a fused matrix op — one instruction performs a full-tile multiply-accumulate: D = A × B + C, the building block of every GEMM
- **How it works**: runs on the Matrix Cores — 4 per CU, one per SIMD, sharing the vector registers
- **Why it matters**: 70–90% of AI compute is GEMM — rocBLAS / MIOpen use MFMA automatically
- Example instruction: `V_MFMA_F32_16x16x16_F16` (FP16 inputs · FP32 accumulate)
- One instruction replaces thousands of scalar FMAs — about 4,000 for a 16×16 tile
- MFMA comes in several tile shapes (e.g., 16×16, 32×32)

## 5. Memory Hierarchy

Bandwidth climbs and latency drops the closer data sits to the CU (fast → slow):

| Level | Capacity | Bandwidth | Latency |
|---|---|---|---|
| Registers (VGPRs) | up to 256 VGPRs/thread | ~100s of TB/s | ~0 cycles |
| LDS (shared memory) | 64 KB per CU | ~10s of TB/s | ~tens of cycles |
| L1 Cache | 32 KB per CU | ~10s of TB/s | ~tens of cycles |
| L2 Cache | 4 MB per XCD (32 MB total) | ~34 TB/s | ~hundreds of cycles |
| Infinity Cache | 256 MB | ~17 TB/s | ~hundreds of cycles (> L2) |
| HBM (global memory) | 192 GB | ~5.3 TB/s | ~hundreds of cycles (slowest) |

**Why it matters:**
- **Bandwidth and latency gap**: on-chip memory delivers far more bandwidth and far lower latency than HBM — roughly 10× or more, registers to HBM
- **The rule of thumb**: VGPRs > LDS > L1 > L2 > Infinity Cache > HBM — keep hot data as close to the CU as possible
- **Reuse > re-fetch**: if a kernel reuses data, hold values in registers through tiling, stage shared data in LDS, or let the caches catch the rest — rather than re-fetching from HBM

*Capacities are MI300X; some bandwidth/latency figures are rough estimates, not measured.*

## 6. GPU Kernels and SPMD

- **Kernel**: a function written to execute on the GPU. When the host program launches the kernel, it creates thousands of threads that process different data in parallel
- **SPMD** (Single Program, Multiple Data): the same kernel runs across many threads at once, each on its own data — it's the model you write in
- Example (vector addition): thread i executes `c[i] = a[i] + b[i]`
- The thread's index picks the data: `i = blockIdx.x * blockDim.x + threadIdx.x`

## 7. Thread Hierarchy — Mapping Threads to the Chip

How the GPU actually runs your code: **Grid → Block → Wavefront → SIMD**

- **Grid**: an array of blocks
- **Block**: up to 1024 threads
- **Wavefront**: 64 threads executed in lockstep on the hardware
- **SIMD**: 16 lanes; 16 lanes per cycle → 64 threads over 4 cycles

## 8. SIMT — The GPU Execution Model

Single Instruction, Multiple Threads — one instruction carried out in lockstep by 64-thread wavefronts:

- **Same instruction, own data**: one instruction is issued to the wavefront; all 64 threads execute it at once — each on its own data, in its own registers
- **Shared program counter**: the 64 threads share one PC, so the wavefront advances in lockstep — one instruction at a time
- **Branches? Hardware masks lanes**: when threads diverge on if/else, the hardware masks inactive lanes automatically
- Scale example: MI300X = 304 CUs × 64 ALUs — **≈20,000 threads executing simultaneously**
- Many resident wavefronts hide memory latency — that's why GPUs want thousands of threads

## 9. CDNA vs RDNA — Two Architectures

- **CDNA**: built for the data center (Instinct — AI & HPC compute)
- **RDNA**: for professional and consumer compute and graphics (Radeon — local AI and graphics)
- Same HIP source, same ROCm stack — from a workstation card to a rack of data center GPUs

## 10. From Workstation to Data Center AI

Scaling AI compute from your desk to the data center, all on the same ROCm stack:

| | Radeon AI Pro R9700 (RDNA 4 · workstation AI) | Instinct MI355X (CDNA 4 · data-center AI & HPC) |
|---|---|---|
| Memory | 32 GB GDDR6 | 288 GB HBM3E |
| Bandwidth | 640 GB/s | 8 TB/s |
| FP16 (matrix) | 191 TFLOPS | 2.5 PFLOPS |
| Scales to | multiple cards over PCIe | 8-GPU nodes · Infinity Fabric |

Per GPU: ≈9× the memory · ≈12× the bandwidth · ≈13× the FP16 throughput — and Infinity Fabric lets a whole node act as one accelerator.

## Module 2 Summary

- **AMD Instinct**: MI100 → MI400 — each generation adds compute, memory, and new precisions, now on an annual cadence. A CU is 4 SIMD units (64 ALUs) + an MFMA engine; memory runs registers → caches → HBM
- **GPU programming and execution model**: your code maps grid → block → wavefront → SIMD; 64 threads run in lockstep (SIMT). Many resident wavefronts hide memory latency — why GPUs want thousands of threads
- **CDNA vs RDNA**: CDNA is built for Instinct data center compute; RDNA powers Radeon for local AI and graphics. Same HIP source, same ROCm stack

---

# Module 3 — ROCm Libraries and AI Frameworks

## 1. Where Libraries Sit in the Stack

Top to bottom:

| Layer | Contents |
|---|---|
| Your Application | PyTorch / JAX / TensorFlow / your code |
| **ROCm Libraries** ★ | rocBLAS, MIOpen, rocFFT … |
| HIP Runtime | C++ runtime API |
| HSA Runtime + amdgpu Driver | GPU resource manager |
| AMD Instinct GPU Hardware | CUs · MFMA engines · HBM · Infinity Fabric |

ROCm compute libraries = optimized building blocks for AI and HPC workloads on AMD GPUs:
**rocBLAS · MIOpen · rocFFT · rocSOLVER · rocRAND · rocSPARSE · rocPRIM** — plus **Triton** for custom kernels.

## 2. Library Overview — What They Do and Where They're Used

| Library | What it does | Where it's used |
|---|---|---|
| rocBLAS / hipBLASLt | Dense linear algebra (e.g., GEMM) | Accounts for much of the computation in modern AI |
| MIOpen | Deep learning primitives (e.g., convolution) | Convolution layers in CNNs |
| rocFFT | Fast Fourier Transforms in 1D, 2D, 3D | Signal processing, spectral methods, physics |
| rocSOLVER | Matrix factorizations and solvers | Scientific computing, PCA, regression |
| rocRAND | Pseudo-random and quasi-random numbers | Simulation, sampling, Monte Carlo |
| rocSPARSE | Sparse matrix and vector operations | Scientific solvers, graph workloads, sparse data |
| rocPRIM / hipCUB | Parallel primitives (scan, sort, …) | Custom kernel building blocks |

## 3. rocBLAS / hipBLASLt — Dense Linear Algebra

The BLAS standard, accelerated. One of the most-called libraries in deep learning.

**What it does:**
- Level 1 (vector–vector): AXPY, DOT, SCAL, NRM2
- Level 2 (matrix–vector): GEMV, TRSV
- Level 3 (matrix–matrix): GEMM, TRSM
- GEMM: `D = α * A * B + β * C` (= A*B when α=1, β=0) — in every linear layer and attention head, and many convolutions

**Why it's fast:**
- Calls the MFMA matrix engine
- Tile sizes are selected per CDNA generation to maximize data reuse in LDS and saturate the MFMA engines
- Software pipeline depth is tuned to hide memory latency behind compute and keep the MFMA units busy

**hipBLASLt**: GEMM-focused, with programmable layouts/types/heuristics, fused bias + activation, and FP8 on CDNA 3+: `Y = Activation(α·A·B + β·C + bias)`

## 4. MIOpen — Deep Learning Primitives

AMD's analog to cuDNN. Used by PyTorch under the hood for conv & batch-norm.

**Operations:**
- Convolution: forward + backward, 2D/3D, multi-algorithm
- Batch norm: training and inference modes
- Pooling: max/average, with optional indices
- Activations: ReLU, GELU (via graph API)
- RNN / LSTM / GRU: fused recurrent cells
- Softmax: fast, accurate, and log-softmax

**Key features:**
- Multi-algorithm dispatch: Winograd · GEMM · direct · implicit-GEMM — picked per shape
- Two-tier autotune: FindDb chooses the algorithm; PerfDb tunes its parameters
- Multi-precision: fp32 · fp16 · bf16 · int8
- Operator fusion: conv + bias + activation in a single kernel

**Why fusion matters:** without fusion, conv → HBM → bn → HBM → relu; with fusion, conv+bn+relu runs as one kernel — reduces off-chip memory access and avoids kernel-launch overhead.

## 5. MIOpen Autotune — Learn Once, Reuse Later

First time: test, tune, and compile. Later: look up and launch.

Three caches:
- **FindDb** (which solution should be used?): remembers the best MIOpen implementation for a convolution problem
- **PerfDb** (how should it be tuned?): remembers low-level kernel tuning choices, such as tile size or block size
- **Kernel cache** (is the GPU code already compiled?): reuses compiled GPU kernels to avoid recompilation

Flow: PyTorch calls `conv2d(x, w)` → build the convolution problem (shape, dtype, layout, conv params) → FindDb hit?
- **Known problem (hit)**: load selected solution from FindDb → load optimized kernel parameters from PerfDb → kernel cache hit? launch directly; otherwise compile the missing kernel and update the cache → launch convolution on GPU
- **New problem (miss)**: try candidate MIOpen solutions and record in FindDb → tune kernels if needed and record optimized parameters in PerfDb → compile, benchmark, update kernel cache → save best solution to FindDb

## 6. rocFFT — Fast Fourier Transforms

Time domain ↔ frequency domain. Backbone of signal processing, physics, spectral solvers.

**Supports:**
- 1D / 2D / 3D, arbitrary lengths
- Real & complex FFTs: R2C / C2R / C2C
- Batched FFTs: process N signals at once
- Precisions: FP16 / FP32 / FP64

**Where you use it:** signal processing (audio, radar, telecom), physics simulations (fluid dynamics, electromagnetics), medical imaging (MRI reconstruction)

## 7. rocSOLVER — Linear Algebra Decompositions

Each decomposition simplifies further calculations.

| Decomposition | Formula | Reveals | Solves |
|---|---|---|---|
| LU | A = L · U | Lower- and upper-triangular factors | General linear systems A·x = b |
| Cholesky | A = L · Lᵀ | For SPD matrices only one triangle needs storing — half the work of LU | Symmetric positive-definite systems, normal equations, GPs |
| QR | A = Q · R | Q has orthonormal columns (QᵀQ = I); stable solving without inverses | Least squares, regression, over-determined systems |
| SVD | A = U · Σ · Vᵀ | Singular values in Σ rank importance; drop the small ones ≈ compression | PCA, low-rank approximation, recommenders, denoising |
| Eigenvalue | A · v = λ · v | Special directions v that A only stretches; λ = stretch factor | Spectral analysis, PageRank, stability, dynamics |

## 8. rocRAND — Pseudo vs Quasi-Random Generators

Two software RNG families with different goals. PRNG and QRNG are both deterministic software generators.

**PRNG — Pseudo-Random**
- Goal: look random and statistically independent enough for simulation
- Deterministic algorithm + seed → reproducible sequence that passes statistical randomness tests
- Point pattern: clustered — has gaps

| Generator | Notes |
|---|---|
| XORWOW | xorshift; fast, simple bit ops |
| MT19937 | Mersenne Twister |
| MTGP32 | MT variant tuned for GPUs |
| Philox | counter-based, parallel-friendly |
| MRG32K3A / MRG31K3P | combined multiple recursive |
| LFSR113 | linear-feedback shift register |
| ThreeFry | counter-based, Threefish cipher |

- Best for: simulation · sampling · randomized algorithms · stochastic ML kernels

**QRNG — Quasi-Random**
- Goal: fill the space uniformly (low-discrepancy)
- Points are constructed to reduce gaps and clumping. No seed like PRNG — indexed by dimension and offset
- Point pattern: uniform — no clumping

| Generator | Notes |
|---|---|
| Sobol32 | 32-bit Sobol low-discrepancy |
| Sobol64 | 64-bit; higher-resolution output |
| Scrambled Sobol32 | scrambled for randomization, still low discrepancy |
| Scrambled Sobol64 | scrambled 64-bit variant |

- All four are based on the Sobol sequence (base-2 partitioning of the unit interval)
- Best for: quasi-Monte Carlo · numerical integration · ray tracing · finance · option pricing
- Note: QRNG = Quasi-Random Number Generators (not Quantum, as in hardware/crypto)

## 9. rocSPARSE — Sparse Matrix Operations

When matrices are mostly zeros — often >99% in graphs, recommenders, and large scientific systems.

**Core ops**: SpMV · SpMM · SpGEMM · SDDMM · sparse solves · format conversion

**Dense vs sparse — a concrete memory example:**
- 10,000 × 10,000 matrix, 0.1% density (typical GNN adjacency):
  - Dense (fp32): 400 MB
  - CSR sparse: ~840 KB
  - → ~475× smaller, and only the non-zeros are touched in compute

**Storage formats:**

| Format | Description |
|---|---|
| CSR | Compressed Sparse Row — most common, fast SpMV |
| CSC | Compressed Sparse Column — column-oriented operations |
| COO | Coordinate list — easy to assemble, sort, and convert |
| ELL | Fixed-width — best when row sizes are uniform |
| BSR | Block Sparse Row — for dense sub-blocks (e.g., FEM) |
| HYB | Hybrid ELL + COO — regular rows in ELL, irregular overflow in COO |

**Applications**: Graph Neural Networks (adjacency matrices are sparse), recommender systems (user × item matrices, mostly empty), scientific solvers (PDEs, FEM, iterative methods), LLM attention (sparse/block-sparse research patterns)

## 10. rocPRIM / hipCUB — Parallel Primitives

Building blocks for whole-array operations and custom kernels. Replaces serial loops with parallel implementations on the GPU.

- **REDUCE**: parallel tree, O(log N) depth (3 levels for 8 elements → log₂N)
- **SCAN** (prefix sum): running totals — each position = the sum so far
- **SORT**: radix/merge — billions of keys/sec on Instinct
- **SELECT** (filter): keep elements where the predicate is true → compacted output (e.g., predicate: x > 3)
- Also: transform · histogram · merge · run-length encoding · partition

## 11. Triton Kernel Language and Compiler on AMD

Python-based kernel DSL for custom and fused GPU kernels.

**Three paths — beyond ROCm libraries:**

| Path | How | Pro | Cost |
|---|---|---|---|
| ROCm libraries | rocBLAS · MIOpen · hipBLASLt; `torch.matmul(a, b)` | Easiest path for standard ops | Limited flexibility for custom fusion |
| Triton | Python DSL · `@triton.jit`; fused, custom, portable kernels | Flexible path for custom kernels | Still requires GPU-kernel thinking |
| Raw HIP | HIP C++ → amdclang/LLVM → AMD GPU binary | Maximum low-level control | Most code and tuning effort |

**Why use Triton:**
1. **Python kernel DSL**: write block-level logic; Triton compiles it for the GPU
2. **Portable across vendors**: same source can target AMD and NVIDIA backends; tuning may differ
3. **Auto-tuning built in**: `@triton.autotune` benchmarks supplied configs and selects the best one per shape
4. **Used by torch.compile**: torch.compile uses TorchInductor; on AMD GPUs Triton is a key GPU codegen path

## 12. PyTorch on ROCm — Familiar Workflows and Ecosystem

Same core PyTorch APIs and AI tools you already work with.

**The full stack (top to bottom):**
1. Ecosystem: HF · vLLM · SGLang · TGI · Lightning
2. PyTorch: torch · autograd · compile
3. ROCm Libraries: rocBLAS · hipBLASLt · MIOpen · RCCL
4. HIP Runtime / HSA Runtime / Driver: HIP · HSA · amdgpu driver
5. Hardware: MI300X · MFMA · HBM · Infinity Fabric

**Common AI workflows:**
- **Train & fine-tune**: single- & multi-GPU training on Instinct; HF Transformers + PEFT; LoRA/QLoRA for LLM fine-tuning; FSDP / DeepSpeed ZeRO / Lightning
- **Serve**: vLLM, SGLang, TGI — OpenAI-compatible APIs; KV-cache management, continuous batching; FP8 GEMM paths on CDNA 3 and newer; large throughput gains vs naive serving
- **Research**: rapid prototyping with standard PyTorch workflows; Lightning, Hydra, W&B, Diffusers, HF

## 13. PyTorch Training Loop — Same Code, Mapped to ROCm

ROCm uses the same torch.cuda API and the 'cuda' device string:

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ROCm uses the torch.cuda API and the 'cuda' device string

model     = resnet18().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**How PyTorch maps down:**

| PyTorch call | What executes underneath |
|---|---|
| `model(inputs)` | MIOpen + rocBLAS/hipBLASLt GEMMs + HIP kernels |
| `loss.backward()` | Autograd → MIOpen + rocBLAS/hipBLASLt GEMMs + HIP kernels |
| `optimizer.step()` | HIP parameter-update kernels (SGD, AdamW, etc.) |
| `inputs.to(device)` | hipMemcpy / hipMemcpyAsync: host DRAM → GPU HBM |
| Any PyTorch op | GPU kernel → grid of thread blocks (workgroups) → wavefronts → execute on CUs |

Familiar PyTorch calls map down to ROCm libraries and AMD GPU execution.

## 14. How torch.compile() Optimizes PyTorch

Captures PyTorch graph regions, fuses compatible operations, and generates optimized code.

- **Eager mode (without torch.compile)**: conv, bias, relu, matmul run as separate ops — separate GPU launches and temporary memory writes → more launches and temporary data
- **With torch.compile**: captures graph regions, fuses eligible ops (e.g., fused bias + relu) → fewer GPU launches and temporary memory writes

**Four optimizations:**
1. **Operator fusion**: compatible ops fused into one generated kernel, reducing intermediate HBM reads/writes
2. **Fewer launches**: each kernel launch has overhead; fewer launches reduce CPU overhead and GPU idle time
3. **Memory planning**: the compiler can reuse temporary storage and avoid some temporary tensor writes
4. **Optional auto-tune**: with mode="max-autotune", Inductor benchmarks candidate Triton configurations and selects the fastest for each shape

One-line change: `model = torch.compile(model)` — default backend = TorchInductor; on AMD it emits Triton kernels and calls ROCm libraries.

## 15. RCCL — Multi-GPU and Multi-Node Communication

AMD's NCCL counterpart for ROCm GPU collectives in torch.distributed, DeepSpeed, and Megatron-LM.

| Collective | Semantics | Typical use |
|---|---|---|
| All-Reduce | Each GPU contributes a same-size array (e.g., element-wise sum); all receive the reduced array | Data-parallel training (gradient sync) |
| All-Gather | Each GPU contributes 1 chunk; all receive the concatenated array | FSDP / ZeRO-3 (gather sharded parameters) |
| Reduce-Scatter | Reduce across GPUs; each receives 1 reduced chunk | FSDP / ZeRO-3 (reduce and shard gradients) |
| Broadcast | One GPU provides an array; all receive the same array | Initial parameter/state sync |
| All-to-All | Each GPU contributes 1 chunk per destination and receives 1 from each destination | MoE token routing (tokens → expert GPU) |

**Hardware mapping** (topology-aware path and algorithm selection):
- Within a node: xGMI / Infinity Fabric or PCIe
- Across nodes: RoCE, InfiniBand, or TCP/IP

## 16. PyTorch and vLLM for Inference on ROCm

PyTorch provides the model framework; vLLM uses it within a specialized LLM inference and serving engine.

**Direct PyTorch inference:**
- General-purpose: LLMs and other model types
- Flexible execution: direct control over model and pipeline
- Custom workflows: preprocessing, logic, and outputs
- Simple deployment: scripts, applications, or services
- → Well suited to general-purpose and custom inference workflows

**vLLM inference & serving:**
- PagedAttention: manages KV cache in page-like blocks
- Continuous batching: dynamically batches active requests
- Quantization: lower precision reduces memory use
- OpenAI-compatible API: integrates with compatible clients
- → Well suited to supported LLMs and high-throughput, concurrent serving

**Run vLLM on ROCm (official image, familiar CLI and API):**

```bash
# Pull the official ROCm vLLM image
docker pull vllm/vllm-openai-rocm:latest

# Start an OpenAI-compatible server
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/model:/model \
  vllm/vllm-openai-rocm:latest \
  --model /model
```

## Module 3 Summary

- **ROCm compute libraries**: use optimized ROCm Compute Libraries before writing a kernel; use Triton for custom kernels the libraries don't cover
- **PyTorch on ROCm**: familiar calls map down to ROCm libraries and HIP; torch.compile() adds Triton fusion
- **Scaling & serving**: RCCL runs multi-GPU/multi-node collectives; for inference, PyTorch directly for flexibility, or vLLM for LLM serving at scale

---

# Module 4 — HIP Programming and Porting from CUDA to HIP

## 1. Module 4 Objectives

- **HIP kernels**: write a HIP kernel from scratch; walk vector-add line by line — `__global__`, thread indexing, the boundary check, and the element-wise add
- **Host & device**: move data across the bus — hipMalloc device buffers, hipMemcpy in and out, launch with `<<<>>>`, synchronize, and check every call
- **CUDA → HIP**: port real CUDA to HIP — run hipify on a histogram kernel; see what swaps automatically and what still needs a human
- **Profile & verify**: time it and prove it works — measure the kernel with HIP events, then verify the output is correct before trusting it

## 2. Reminder — What HIP Is, and Why We Use It

A compiler that targets both AMD and NVIDIA GPUs from a single source file:

- Flow: `my_kernel.hip` → hipcc → AMD GPU (via amdclang → AMD ISA) or NVIDIA GPU (via nvcc → NVIDIA PTX)
- **Same source**: the same .hip file compiles on either platform — no #ifdefs, no per-vendor branches
- **CUDA-like syntax**: `__global__`, `<<<grid, block>>>`, threadIdx, blockIdx — all the familiar CUDA primitives
- **Most CUDA code ports**: run hipify-perl once to rename Cuda → Hip; most kernels compile immediately

## 3. Host vs Device — Two Worlds, Two Memories

Two separate machines, two separate memories. The vectors live on both sides; hipMemcpy is the only bridge.

- Host (CPU + DRAM) ⇄ Device (Instinct GPU: Compute Units + HBM)
- **Separate address spaces**: CPU and GPU each have their own memory. A host pointer is meaningless on the device. Keep two pointers per vector: h_X on the host, d_X on the device
- **Explicit data transfer**: nothing moves on its own. hipMemcpy carries the inputs to HBM before the launch and the result back after
- **Perfectly parallel example**: C[i] = A[i] + B[i] is independent per element. N = 2^20 elements map to ~1M threads; no thread reads another's value — the ideal GPU workload

## 4. Vector Add — The Schema

**C[i] = A[i] + B[i]**: three vectors, one addition per element — and one GPU thread per element.

- **One thread, one element**: thread i reads A[i] and B[i] and writes C[i]; its index is `i = blockDim.x * blockIdx.x + threadIdx.x`
- **Fully independent**: no element depends on any other, so all N additions can run at the same time — the ideal GPU workload
- **No loop**: the CPU's for-loop over N is replaced by launching N threads; the hardware does the iteration for you

## 5. Vector Add — The Host Workflow, Step by Step

| Step | What | API | Where |
|---|---|---|---|
| 1. Allocate device memory | Reserve d_A, d_B, d_C in GPU memory | `hipMalloc(&d_A, bytes)` ×3 | on GPU |
| 2. Copy the inputs in | Push A and B to the device | `hipMemcpy(d_A, h_A, bytes, HostToDevice)` | CPU → GPU |
| 3. Launch the kernel | Run parallel: one thread per element C[i]=A[i]+B[i] | `vec_add<<<(N+255)/256, 256>>>(…)` | on GPU |
| 4. Check & synchronize | Catch a bad launch, then wait for the kernel | `hipGetLastError(); hipDeviceSynchronize()` | on GPU |
| 5. Copy the result out | Bring C back to host memory once it is done | `hipMemcpy(h_C, d_C, bytes, DeviceToHost)` | GPU → CPU |
| 6. Verify & free | Confirm the output, then release the buffers | `check h_C[]` → `hipFree(d_A/d_B/d_C)` | on CPU |

## 6. Key Building Blocks

**`__global__` — the kernel qualifier**
- Marks the function as a GPU kernel: compiled for the device, called from the host with `<<<...>>>`, run by many threads in parallel
- Always returns void — results flow out through the C pointer
- Example: `__global__ void vec_add(const float* A, const float* B, float* C, int N)`
- Cousins: `__device__` for GPU-only helpers, `__host__` for CPU code (the default)

**Boundary check — don't step out of bounds**
- Ceiling division `(N + 255) / 256` rounds the block count up, so the last block may launch threads with i ≥ N
- `if (i < N)` makes those surplus threads do nothing — without it they read and write past the ends of A, B, and C

**hipMemcpy — crossing the PCIe bridge**
- `hipMemcpy(dst, src, bytes, direction)`
- HostToDevice carries h_A, h_B into d_A, d_B before the launch
- DeviceToHost carries d_C back into h_C afterward
- It is synchronous — the copy back also waits for the kernel to finish

```c
HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
// ...kernel...
HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
```

**Launch + synchronize — `<<<grid, block>>>`**
- Dispatch ~1M threads, then confirm the launch succeeded and finished:

```c
vec_add<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
HIP_CHECK(hipGetLastError());          // catch bad launch config
HIP_CHECK(hipDeviceSynchronize());     // catch runtime kernel errors
```

- `<<<(N+255)/256, 256>>>` queues the kernel with 256 threads/block (4 wavefronts) and returns immediately
- `hipGetLastError()` catches a bad launch configuration; `hipDeviceSynchronize()` blocks until the GPU finishes and surfaces runtime faults

## 7. hipify — perl vs clang

Porting is mostly renaming cuda* → hip* across many files; hipify automates it. Two tools — know which to reach for:

| | hipify-perl | hipify-clang |
|---|---|---|
| Nature | Regex find-and-replace. No compiler — instant and scriptable | Clang AST translator. Parses the code like a real compiler |
| What it is | Renames known CUDA tokens to HIP by text pattern; doesn't parse the code | Understands the real C++ structure, so it rewrites types a regex can't see |
| Handles | cuda* → hip* names · the #include swap · known enums and types · `<<<...>>>` launches | Everything perl does, plus templates · typedef/macro-hidden types · header chasing |
| Use when | A single, self-contained file | Large codebases, templated kernels, or library-heavy ports |

## 8. Practice: Port the Histogram (CUDA → HIP)

hipify-perl swaps the prefixes; the kernel is identical. Convert, build with hipcc, verify the counts.

**Kernel (identical on both sides)** — `__global__`, `__shared__`, `__syncthreads()`, threadIdx/blockIdx/blockDim/gridDim, atomicAdd all unchanged:

```c
#include <hip/hip_runtime.h>       // was: <cuda_runtime.h>
#define NUM_BINS 256
__global__ void histogram(const unsigned char* data, int* bins, int N) {
    __shared__ int local_bins[NUM_BINS];
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        local_bins[i] = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
        atomicAdd(&local_bins[data[i]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        atomicAdd(&bins[i], local_bins[i]);
}
```

**Host code (only prefixes change)**: cudaMalloc→hipMalloc, cudaMemset→hipMemset, cudaMemcpy→hipMemcpy, cudaDeviceSynchronize→hipDeviceSynchronize, cudaFree→hipFree, cudaMemcpyHostToDevice→hipMemcpyHostToDevice.

**Full flow:**

```bash
$ hipify-perl histogram_cuda.cu -o histogram_hip.cpp    # cuda* → hip*
$ hipcc -o histogram_hip histogram_hip.cpp && ./histogram_hip
  bins[0]=0, bins[1]=1, bins[2]=2, ... bins[255]=255    ✓ ascending counts verified
```

## 9. Module 4 Essential Questions

- **Q1: What does the `__global__` keyword mark, and where does that function run?** It marks a kernel: a function launched from the host (CPU) but executed on the device (GPU), with each thread running one copy.
- **Q2: In a HIP program, how do the host and the device divide the work?** The host (CPU) runs the main program: it allocates memory, moves data, and launches kernels. The device (GPU) runs those kernels and does the heavy parallel computation.
- **Q3: What is hipify, and why does it make porting from CUDA so cheap?** hipify is a tool that automatically rewrites CUDA API calls into their HIP equivalents.

## Module 4 Summary

- **Host vs device**: two separate memories. hipMalloc device buffers, keep h_X and d_X pointers, and hipMemcpy across the PCIe bus both ways
- **The kernel**: `__global__` runs on the GPU. Each thread handles one element: idx = blockIdx.x·blockDim.x + threadIdx.x
- **CUDA → HIP**: hipify-perl swaps cuda* → hip*; the kernel is unchanged. Use hipify-clang only for templated or macro-hidden types

---

# Module 5 — Performance Optimization: Measure · Tune · Validate

## 1. The Roofline Model — Pick the Right Fix

Two ceilings bound any kernel; arithmetic intensity decides which one you hit — and that picks the family of fixes.

**Core formula: `P = min(π, β·I)`**
- P: attainable performance (FLOP/s)
- π: peak compute (FLOP/s — GPU spec)
- β: peak memory bandwidth (GPU spec)
- I: arithmetic intensity = the kernel's FLOPs ÷ bytes it moves (FLOP/byte)
- Ridge: P = π/β — where the memory roof meets the compute roof

**Worked example — SAXPY (y = a·x + y):**
- 2 FLOPs = one multiply (a·x) + one add (+y)
- 12 bytes = read x + read y + write y, 4 B each (scalar a stays in a register — not counted)
- I = 2 ÷ 12 ≈ 0.17 FLOP/byte → far left of the ridge, memory-bound

**Fix families by side of the ridge:**
- **Left of ridge — memory-bound**: move fewer bytes, reuse more — coalescing, LDS tiling, fusion, lower precision
- **Right of ridge — compute-bound**: use the ALUs better — Matrix Cores, mixed precision, more ILP, libraries

**RX 7900 XT reference**: β ≈ 0.8 TB/s, FP32 π ≈ 51 TFLOPS → ridge ≈ 64 FLOP/byte. The FP16/BF16 WMMA roof (~103 TFLOPS) sits higher, pushing the ridge right — which is why GEMM loves half precision.

## 2. Method Navigator — From Diagnosis to Fix

Profile, place the kernel on the roofline, then read down its column:

```bash
$ rocprofv3 --stats --kernel-trace -- ./app   # build the roofline → which roof are you hitting?
```

| Bottleneck | Symptoms (tell) | Fixes |
|---|---|---|
| **Memory-bound** | MemUnitBusy high · ALUs idle · low AI | Coalesced access (M2) · LDS tiling — transpose & reuse (M2) · Bank-conflict padding (M2) · Kernel fusion to raise AI (M3) |
| **Compute-bound** | VALUBusy / MFMA near peak · memory idle | MFMA matrix cores (M2) · Mixed precision FP16/BF16 (M5) · rocBLAS / MIOpen (M3) |
| **Overhead-bound** | Neither roof reached · GPU starved | Block size (M1b) · Occupancy / latency hiding (M1b) |

**Other levers (from the HIP perf guide, not exercised in the lab):**
- Grid-stride loop — fewer fixed blocks; each thread loops over many elements — amortizes dispatcher cost (overhead-bound)
- Loop unrolling — several elements per iteration cut control overhead; over-unrolling spills registers (compute-bound)
- Cut divergence — keep all lanes on one branch; a split runs both sides with idle lanes masked off (compute-bound)
- Parallel reduction — fold sums/maxes pairwise through LDS in log₂n steps, not a serial accumulator (compute-bound)

## 3. Choosing the Right Method — The Decision Tree

Walk the tree: profile, rule out overhead first, then read which roof you're against — and fix only the binding limit.

1. **Profile**: `rocprofv3 --stats`
2. **Is the GPU busy? (a roof near peak?)**
   - **No — idle** → **OVERHEAD-BOUND**. Tell: both counters low, GPU idle, tiny/many launches. Fix: grid-stride, batch, bigger and fewer launches
   - **Yes** → 3. **Which roof is saturated?**
     - **Memory roof** → **MEMORY-BOUND**. Tell: MemUnitBusy high, ALUs idle, low arithmetic intensity (few FLOPs/byte). Fix: coalesce, LDS tiling, fusion, fewer bytes
     - **Compute roof** → **COMPUTE-BOUND**. Tell: VALUBusy/MFMA/WMMA high, memory idle. Fix: Matrix Cores, mixed precision, libraries

## 4. The Module 5 Lab: Three Experiments

Now we apply the recapped techniques to real code:
1. **Experiment 1** — Matrix transpose: coalescing & LDS tiling → MEMORY-bound fix
2. **Experiment 2** — SAXPY: block size & occupancy → OVERHEAD-class fix
3. **Experiment 3** — PyTorch GEMM: FP16/BF16 mixed precision → COMPUTE-bound fix

### Experiment 1 — Matrix Transpose: Coalescing & LDS Tiling (MEMORY-bound)

**The kernel**: a matrix transpose — flip across the diagonal, so (row, col) moves to (col, row): out[x][y] = in[y][x]. **The catch**: reading the input runs along rows (neighbouring, contiguous), but writing the output runs down columns (strided, far apart). That read/write mismatch is the whole performance story.

**Baseline (naive)** — each thread copies one element to its transposed slot; one element per thread, no shared memory:

```c
__global__ void transpose_naive(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H)
        out[x * H + y] = in[y * W + x];
        // ① read in[y*W+x]  → contiguous: consecutive threads hit consecutive
        //    addresses → one coalesced memory burst. Fast.
        // ② write out[x*H+y] → stride H: neighbouring threads write far apart,
        //    the GPU can't combine them — each write is its own slow trip.
}
```

Same FLOPs as the tiled fix — only the write pattern differs; that's what the profiler flags.

**Optimized (tiled)** — same kernel, four marked changes. Stage a tile in LDS, sync, then write it back transposed — both ends coalesced:

```c
#define TILE 16
__global__ void transpose_tiled(const float* in, float* out, int W, int H) {
    __shared__ float tile[TILE][TILE + 1];          // ① +1 pad
    int xIn = blockIdx.x * TILE + threadIdx.x;
    int yIn = blockIdx.y * TILE + threadIdx.y;
    if (xIn < W && yIn < H)
        tile[threadIdx.y][threadIdx.x] = in[yIn * W + xIn];   // ② coalesced load
    __syncthreads();                                           // ③ fence tile
    int xOut = blockIdx.y * TILE + threadIdx.x;
    int yOut = blockIdx.x * TILE + threadIdx.y;
    if (xOut < H && yOut < W)
        out[yOut * H + xOut] = tile[threadIdx.x][threadIdx.y]; // ④ coalesced store
}
```

What changed: ① +1 padding shifts each row a bank → conflict-free transposed reads; ② load coalesced — neighbouring lanes, neighbouring memory addresses; ③ one sync fences the tile before any thread reads it back; ④ store coalesced too — this is the write that was strided in the baseline; now it is contiguous, which is the whole point of the fix.

**Profile & results (Radeon RX 7900 XT, 4096² transpose):**

```bash
$ hipcc -O3 -o transpose transpose.cpp && ./transpose
$ rocprofv3 --stats --kernel-trace -T -o profiler/lab5/transpose -- ./transpose
  # → transpose_kernel_trace.csv + transpose_*_stats.csv
```

| Kernel | Time/call | Eff. bandwidth | vs naive |
|---|---|---|---|
| transpose_naive (uncoalesced) | 1.272 ms | 105.5 GB/s | 1.0× |
| transpose_tiled (LDS) | 0.260 ms | 515.8 GB/s | **~4.9×** |

Takeaways: identical work, yet naive trails 105.5 vs 515.8 GB/s — the profiler shows high memory-unit busy, memory-bound on the strided write. LDS staging coalesces the write: 1.272 → 0.260 ms. Both sit under the ~0.8 TB/s memory peak — tiled reaches ~64%, naive only ~13%. GB/s vs peak is the real signal, not time alone.

### Experiment 2 — SAXPY: Occupancy & Block Size (OVERHEAD class)

**The catch**: SAXPY's arithmetic intensity is only I ≈ 0.17 FLOP/byte — tiny, so SAXPY sits far left of the ridge and is firmly memory-bound. Compute is never the limit — so the fix is an occupancy/latency-hiding lever: enough resident wavefronts to reach the bandwidth ceiling.

```c
// SAXPY: y = a*x + y  (memory-bound)
__global__ void saxpy(float a, const float* x, float* y, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // ① 1 elem/thread
    if (i < N)
        y[i] = a * x[i] + y[i];                       // ② 2 reads, 1 write
}
// host: N = 1<<24, time each size ×50
int sizes[] = {64, 128, 256, 512, 1024};              // ③ sweep block size
for (int bs : sizes)
    saxpy<<<(N + bs - 1) / bs, bs>>>(2.0f, d_x, d_y, N);
```

What to notice: ① one element per thread — the kernel is trivial; the launch shape is what we vary; ② two reads + one write — almost no math per byte, so SAXPY is firmly memory-bound (very low AI); ③ sweeping five block sizes — bigger blocks keep more wavefronts resident, which hides memory latency — up to the point bandwidth saturates.

**Results (Radeon RX 7900; effective bandwidth vs the ~0.8 TB/s GDDR6 peak):**

| Block size | Eff. bandwidth |
|---|---|
| 64 | 657.5 GB/s (slowest — can't field enough wavefronts to hide latency) |
| 128 | 735.2 GB/s |
| 256 | 736.4 GB/s |
| 512 | 739.2 GB/s |
| 1024 | 742.4 GB/s (best) |

Why it plateaus: from 128 up the kernel is bandwidth-bound — all sit at ~740 GB/s, about 93% of the 0.8 TB/s peak. Block size barely matters once memory is saturated. **Saturation**: unlike a compute-bound kernel, SAXPY can't beat its memory ceiling — the best you can do is reach it. Here 128–1024 threads all do.

### Experiment 3 — PyTorch GEMM: FP16/BF16 Mixed Precision (COMPUTE-bound)

**Theory**: half-precision moves half the bytes and feeds the Matrix Cores. It is faster — but it is a different, lower-precision computation.

| Format | Bits | Exponent | Mantissa | ~Digits |
|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ~7 |
| FP16 | 16 | 5 | 10 | ~3–4 |
| BF16 | 16 | 8 | 7 | ~2–3 |

- FP16 keeps more mantissa (precision); BF16 keeps FP32's 8-bit exponent, so it rarely overflows or underflows
- **The rule**: each result in a matrix multiply adds up thousands of products. Adding them in FP16 loses accuracy, so the inputs stay FP16/BF16 but the running total is kept in FP32
- **Matrix units**: Radeon (RDNA3) uses WMMA; Instinct/CDNA use MFMA — same idea (half-precision in, FP32 accumulate), different instruction. RDNA3 has no FP32 matrix path
- **Why faster**: half the bytes through memory and LDS, plus the WMMA matrix cores — higher throughput at the same shape
- **The tradeoff**: not a free 2×; it is lower-precision math. Verify the result is accurate enough for the application

**Benchmark code (PyTorch; the marked lines are all that change between FP32 and FP16/BF16):**

```python
import torch, time
device = torch.device('cuda')     # 'cuda' = HIP alias
M, N, K = 2048, 2048, 2048

def bench_gemm(dtype, label):                       # ① dtype is the only change
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    torch.mm(A, B); torch.cuda.synchronize()        # ② warm-up: excludes one-time
    t0 = time.perf_counter()                        #    init/autotune from timing
    for _ in range(50):
        C = torch.mm(A, B)                          # ③ → rocBLAS under the hood,
    torch.cuda.synchronize()                        #    no kernel to write
    ms = (time.perf_counter() - t0) / 50 * 1000
    gflops = 2*M*N*K / (ms*1e-3) / 1e9              # ④ GFLOPS = 2·M·N·K/time —
    print(f"{label}: {ms:.3f} ms  {gflops:.1f} GFLOPS")  # fair, size-independent metric

bench_gemm(torch.float32,  "FP32")
bench_gemm(torch.float16,  "FP16")
bench_gemm(torch.bfloat16, "BF16")
```

**Results (2048³ GEMM, Radeon RX 7900 XT):**

| Precision | Time/call | GFLOPS | Speedup |
|---|---|---|---|
| FP32 (baseline) | 6.645 ms | 2,585 | 1.0× |
| FP16 | 0.291 ms | 59,135 | **22.9×** |
| BF16 | 0.289 ms | 59,513 | **23.0×** |

- FP16 — 22.9×: runs on the WMMA matrix units. 5-bit exponent — watch overflow on long sums
- BF16 — 23.0×: also WMMA; FP32's 8-bit exponent range — safer when values span a wide range
- **Why FP32 is so slow**: RDNA3 has no FP32 matrix path, so FP32 GEMM falls back to the vector ALUs (~2.6 TFLOPS). FP16/BF16 use WMMA (~59 TFLOPS) — that's the 23×

## 5. Lab Results — What We Measured

The three experiments, side by side — all values measured on a Radeon RX 7900. Measure first, fix the bottleneck, re-measure.

| Experiment | What changed | Result | Status |
|---|---|---|---|
| Transpose — naive | uncoalesced strided write | 105.5 GB/s | baseline |
| Transpose — tiled (LDS) | stage 16×16 tile, coalesced | 515.8 GB/s (4.9×) | fix |
| SAXPY — block size & occupancy | 128+ threads saturate | ~742 GB/s (best) | tuned |
| GEMM — FP16/BF16 | half precision via rocBLAS | 22.9× / 23.0× GFLOPS | trade-off |

**Three principles:**
1. **Measure first**: profiler, not intuition. Effective bandwidth and GB/s vs the ~0.8 TB/s GDDR6 peak name the bottleneck
2. **Access pattern wins**: coalescing + an LDS tile beat extra compute — same FLOPs, just far better effective bandwidth
3. **Right-size & reuse libs**: 128+-thread blocks saturate bandwidth; FP16/BF16 via rocBLAS for throughput — then verify accuracy

## 6. Module 5 Essential Questions

- **Q1: In the GEMM experiment we ran FP16/BF16 — what does "mixed precision" actually mean, and why is the accumulator still FP32?** Keep the inputs in FP16/BF16 (half the bytes, fed straight to the matrix units) but add up the products in FP32. The wider FP32 accumulator keeps the long running sum accurate — half-precision speed without wrecking the result.
- **Q2: You profile a kernel: the GPU is busy, but the ALUs sit idle while the memory units are saturated. Which roof are you on and what fix does it point to?** Memory-bound — you are on the memory roof (low arithmetic intensity). The tree points to memory fixes: coalesce accesses, stage data in LDS, fuse kernels — move fewer bytes or reuse them. More compute would not help.
- **Q3: On the Method Navigator, which bottleneck class do coalescing and LDS tiling belong to?** Coalescing and LDS tiling are memory-bound fixes — they cut or speed up byte traffic.

---

# Quick-Reference: Numbers Worth Memorizing

| Fact | Value |
|---|---|
| MI300X CUs | 304 active (38 per XCD × 8 XCDs) |
| CU composition | 4 SIMDs × 16 lanes = 64 ALUs + MFMA + LDS 64 KB + L1 32 KB |
| Wavefront | 64 threads in lockstep; executes as 16 lanes/cycle over 4 cycles |
| Block limit | up to 1024 threads |
| MI300X memory | 192 GB HBM3 (~5.3 TB/s) · 256 MB Infinity Cache · 32 MB L2 total |
| GEMM share of AI compute | 70–90% |
| Ceiling-division launch idiom | `<<<(N+255)/256, 256>>>` (256 threads = 4 wavefronts) |
| Roofline | P = min(π, β·I); SAXPY I ≈ 0.17 FLOP/byte |
| Lab results | Transpose LDS tiling 4.9× · SAXPY saturates ≥128 threads (~93% peak) · GEMM FP16/BF16 ~23× |

*End of study notes — covers Modules 2–5 of the ROCm Certification Level 1 course.*
