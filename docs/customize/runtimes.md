# Runtimes

## What is it?

A **runtime** is the engine that actually serves a model. Three are supported:

| Runtime | Use case | Key optimizations |
|---------|----------|-------------------|
| **vLLM** | LLMs, VLMs | NUMA auto-bind, AMX kernels, prefix caching, chunked prefill |
| **OpenVINO (OVMS)** | Embedding, reranking, INT4/INT8 LLMs | OpenVINO IR, MediaPipe graph |
| **vLLM XPU** (`vllm-xpu`) | LLMs on Intel Arc Pro B-Series (XPU) GPUs | PagedAttention; single-GPU models for now |

Each engine is configured **once**, in the `runtimes:` section at the bottom of
`models.yaml`. That is where you define the available versions and their container
images, the settings shared by every model on that engine, and the per-workload
(category) defaults — CPU, memory, engine flags, and env vars. Keeping it in one place
means the full picture of what a server runs with is easy to see and to change.

Models then opt into a runtime and version through their
[server binding](catalog.md#choose-which-server-and-version-a-model-may-use).

## How settings combine

Every level overrides the one before it:

```
env   = runtime common env  +  category env  +  version env delta  +  per-model env  +  --env
args  = category args       +  version args delta  +  per-model args  +  --arg
image = the selected version's image                 (or per-model `image:`)
```

Version selection, highest precedence first:

```
--server-version   >   the model's server_version:   >   the runtime's default_version
```

## vLLM

A **versioned** runtime. Adding a new engine version is a new entry under `versions:` —
no code changes.

```yaml
runtimes:
  vllm:
    default_version: "0.24.0"
    env: { ... }                 # applied to every vLLM pod
    versions:
      "0.24.0":
        image: "docker.io/vllm/vllm-openai-cpu:v0.24.0"
      "0.28.0":                  # only models that pin it use it (qwen3-6-35b-a3b)
        image: "docker.io/vllm/vllm-openai-cpu:v0.28.0"
      "0.19.1":
        image: "docker.io/vllm/vllm-openai-cpu:v0.19.1"
        # optional env:/args: DELTAS on top of the common env and category args
    categories:
      llm:    { defaults: { cpu: 8,  memory: "16Gi", kv_cache_gb: 8 } }
      embed:  { defaults: { cpu: 6,  memory: "16Gi", kv_cache_gb: 16 } }
      rerank: { defaults: { cpu: 6,  memory: "16Gi", kv_cache_gb: 16 } }
      vlm:    { defaults: { cpu: 16, memory: "48Gi", kv_cache_gb: 8 } }
```

**Adding a version:** copy an existing entry, change the key and `image`, deploy a model
against it with `--server-version`, then promote it by moving `default_version`.

> [!WARNING]
> Per-version `args:` deltas must suit the models that will use them. MoE-only flags
> such as `--enable-expert-parallel` **crash dense models**.

### CPU KV cache

`VLLM_CPU_KVCACHE_SPACE` is deliberately **not** in the shared `env:` — it is set
per-category from that category's `kv_cache_gb`, so each workload class sizes its CPU KV
cache appropriately.

`embed` and `rerank` do no autoregressive generation, but vLLM CPU still sizes a KV cache
at startup: without the variable it needs most of the NUMA node's memory free, and crashes
when an LLM already runs on that node. So they set it too, from `kv_cache_gb: 16`.

### Plumbing

Low-level, stable settings live in `model_manager/runtimes/vllm/values.yaml` and are
merged in at load time. `models.yaml` wins on any key clash.

```yaml
device: cpu
kind: LLMInferenceService
routing: epp
```

## OpenVINO (OVMS)

A **flat** runtime — there is no version axis. `server_version` does not apply, and
requesting one with `--server-version` aborts with a clear message.

```yaml
runtimes:
  openvino:
    env: {}
    categories:
      llm:    { defaults: { cpu: 8, memory: "16Gi" } }
      embed:  { defaults: { cpu: 6, memory: "12Gi" } }
      rerank: { defaults: { cpu: 6, memory: "12Gi" } }
```

Plumbing in `model_manager/runtimes/openvino/values.yaml`:

```yaml
image: "openvino/model_server:2026.4.1"                   # serving and weights download
device: cpu
kind: InferenceService
format: openvino
runtime_name: openvino-runtime                            # ClusterServingRuntime to reference
routing: aigateway
serving_runtime_manifest: openvino/serving-runtime.yaml   # applied on first deploy
```

### `weight_format` — only for raw Hugging Face repos

`weight_format` is a **conversion** knob for `ovms --pull`, set per category. Use it
(`int8`, `int4`, `fp16`, …) only when `model_id` points at a raw Hugging Face repo that
must be converted and quantized to OpenVINO IR.

> [!IMPORTANT]
> Leave it **unset** for models already published as OpenVINO IR (the `OpenVINO/*`
> repos). Passing `--weight-format` to an already-converted model makes `ovms --pull`
> abort with *"unsupported for OpenVINO models"*.

### API paths

Clients always use the OpenAI `/v1/` paths. For OpenVINO-served models the AI gateway
rewrites `/v1/` → `/v3/` internally, since OVMS uses the v3 API — this is invisible to
callers. See [Inference Request Flow](../reference/request_flow.md).

## vLLM XPU (Intel Arc Pro B-Series XPU)

vLLM's XPU build, as a runtime of its own (`vllm-xpu`) rather than a version of the CPU
`vllm` runtime: it ships as a separate image and runs as a plain `InferenceService` — one
pod per model on the node GPU(s) it requests — instead of an `LLMInferenceService`.
A **flat** runtime, like OpenVINO — no version axis.

```yaml
runtimes:
  vllm-xpu:
    env:
      VLLM_WORKER_MULTIPROC_METHOD: "spawn"   # XPU contexts cannot survive fork()
      TORCH_LLM_ALLREDUCE: "1"
      CCL_ZE_IPC_EXCHANGE: "pidfd"
      # ... telemetry off, SYCL kernel cache, oneCCL ring all-reduce — see models.yaml
    categories:
      llm:
        args: [ --host, 0.0.0.0, --port, "8000", --dtype, float16, --block-size, "64",
                --gpu-memory-utilization, "0.9", --max-model-len, "32768",
                --enforce-eager, --disable-prefix-caching, --trust-remote-code ]
        defaults: { cpu: 8, memory: "64Gi", gpu: 4 }
```

Plumbing in `model_manager/runtimes/vllm-xpu/values.yaml`:

```yaml
image: "vllm/vllm-openai-xpu:v0.33.0"
device: gpu
kind: InferenceService
format: vllm-xpu
runtime_name: vllm-xpu-runtime                              # ClusterServingRuntime to reference
routing: aigateway
serving_runtime_manifest: vllm-xpu/serving-runtime.yaml     # applied on first deploy
accelerator_resource: "gpu.intel.com/xe"                    # extended resource the Intel GPU device plugin advertises
```

Weights are downloaded to the model PVC as for the CPU runtime and passed as
`--model /models/<name>`. Per-model `args` override the category's (last flag wins),
e.g. `--dtype bfloat16` for Llama.

> [!NOTE]
> Unlike the Helm-based XPU path of the Agent Toolkit 1.1 release, no host driver libraries
> are mounted: the image's own Level Zero stack drives Arc Pro B70 cards, and the pod runs
> with every Linux capability dropped.

### `gpu` — accelerators per model

Unlike the CPU runtimes, `vllm-xpu` models also set `gpu` (per model or category
`defaults`): the number of accelerators requested via `accelerator_resource` in the
container's `resources.limits` — one GPU per tensor-parallel rank, so keep it equal to the
model's `--tensor-parallel-size` (the unit tests enforce this for the shipped catalog):

```yaml
- name: vllm-xpu-gpt-oss-20b
  model_id: openai/gpt-oss-20b
  runtime: vllm-xpu
  gpu: 1
  args: [ "--tensor-parallel-size", "1", "--dtype", "auto", "--disable-sliding-window",
          "--max-model-len", "9216", "--max-num-batched-tokens", "9216" ]
```

`cpu`/`memory` still apply — they size the host-side process that drives the
accelerators, not the accelerators themselves.

### Prerequisites

The cluster needs the [Intel GPU device plugin](https://github.com/intel/intel-device-plugins-for-kubernetes)
deployed, so nodes advertise `gpu.intel.com/xe`; Kubernetes schedules a model's pod onto a
node with enough free accelerators purely from the extended-resource request, no further
node affinity needed.

### CPU policy

GPU/XPU runtimes (`device: gpu` in the runtime's `values.yaml`) are deployed with the
`best-effort` CPU policy by default — NUMA-aware CPU pinning targets CPU-bound serving,
and buys nothing for a process that is bound by its accelerators. Pass `--cpu-policy`
explicitly to override.

## Which kind of Kubernetes object?

The runtime's `kind` decides what `model-manager` creates:

| Runtime | Object | Template |
|---|---|---|
| vLLM | `LLMInferenceService` | `model_manager/templates/llm-inference-service.yaml` |
| OpenVINO | `InferenceService` | `model_manager/templates/inference-service.yaml` |
| vLLM XPU | `InferenceService` | `model_manager/templates/inference-service-vllm-xpu.yaml` |

The raw `ClusterServingRuntime` manifests stay as files under
`model_manager/runtimes/<name>/`.
