# KV-cache offload (KVShrink)

## What is it?

The `vllm-cuda-kvshrink` runtime is `vllm-cuda` with [KVShrink / IAXL](https://github.com/intel/intel-accel-for-llm)
built in. vLLM spills KV-cache blocks from GPU memory into a DDR pool and reloads them when a request reuses the
prefix, so a GPU node holds longer contexts or more concurrent requests. The pool is either:

- **local**: in the vLLM pod, on the GPU node, or
- **remote**: in a pool daemon on another node, which the vLLM pod reaches over RDMA (RoCE). The pool node
  compresses the blocks, with the CPU or with Intel QAT and IAA.

A model opts in with `kv_offload:` in `models.yaml`; model-manager renders the pool settings, sizes the pod's
memory, and in remote mode creates, waits for and removes the `<model>-kv-pool` daemon with the model.

## Prerequisites

| | Local pool | Remote pool |
|---|---|---|
| GPU node | NVIDIA GPU, `nvidia.com/gpu` advertised | the same, plus an RDMA NIC |
| Pool node | — | RDMA NIC that routes to the GPU node's; for QAT/IAA, Intel QAT (in-tree `qat_4xxx`, Linux 6.0+) and IAA with PASID enabled |
| Platform components | — | `accel_device_plugins` with `rdma` on both nodes (`rdma/hca`) and, for QAT/IAA, `qat`/`iaa` on the pool node (`qat.intel.com/dc`, `iaa.intel.com/wq-user-shared`) — see the solutions repo's `docs/deploy/install_platform.md` |
| Namespace | any | no `baseline`/`restricted` Pod Security enforcement: both pods run on the host network, because UCX binds the NIC that owns the RDMA IP. With `enforce_pss: true`, deploy the model to another namespace |
| Container runtime | memlock unlimited (kubespray's default `containerd_limit_mem_lock: infinity`) | the same; for QAT/IAA, `device_ownership_from_security_context: true` (kubespray `containerd_extra_runtime_args`) so the non-root daemon can open the VFs and work queues |

RDMA itself (MLNX_OFED / DOCA-OFED, port up, RoCE v2 GIDs) is set up on the hosts beforehand.

## Build the image

There is no public image. Build `model_manager/runtimes/vllm-cuda-kvshrink/docker/` for the GPU node's kernel and
push it to a registry the nodes pull from:

```bash
cd model_manager/runtimes/vllm-cuda-kvshrink/docker
docker build --build-arg KERNEL_HEADERS_PACKAGE=linux-headers-<GPU node's uname -r> \
  -t <registry>/vllm-kvshrink:v0.29.0-556c832 .
docker push <registry>/vllm-kvshrink:v0.29.0-556c832
```

Then set it in `models.yaml`:

```yaml
runtimes:
  vllm-cuda-kvshrink:
    versions:
      "0.29.0":
        image: "<registry>/vllm-kvshrink:v0.29.0-556c832"
```

IAXL and vLLM must be a matched pair: IAXL main `556c832` (the Dockerfile's `KVSHRINK_GIT_REF`) targets vLLM 0.29.0
(`BASE_IMAGE`). On a mismatch the engine dies on the first prompt longer than one KV block with
`AssertionError: all GPU tensors must be contiguous`. Change both together, and the version key with them.

## Configure a model

```yaml
- name: nemotron-mini-4b-kvshrink
  model_id: nvidia/Nemotron-Mini-4B-Instruct
  runtime: vllm-cuda-kvshrink
  gpus: 1
  kv_offload:
    pool_gb: 256
```

| Key | Default | Meaning |
|---|---|---|
| `pool_gb` | required | DDR pool per TP rank (`IAXL_DDR_POOL_SIZE_GB`) |
| `scratch_gb` | 8 | scratch pool per rank (`IAXL_SCRATCH_POOL_SIZE_GB`) |
| `compression` | `cpu` | `cpu` (zlib), `off` (blocks stored raw), `qat`, `iaa` or `qat+iaa` |
| `qat_vfs` | required with QAT | QAT VFs per rank (`QAT_POLICY`), at most one per QAT PF |
| `qat_devices` | `0..qat_vfs-1` | `KVSHRINK_QAT_DEVICES`, `\|`-separated per rank; required when TP > 1 |
| `iaa_wqs` | required with IAA | IAA shared work queues, one per IAA device: IAXL uses every device on the node |
| `zip_threads` | QAT/IAA: vCPUs − 4 | `IAXL_OMP_THREAD_NUM`, threads driving the compressors |
| `cpu_zip_threads` | 4 (IAXL's) | `IAXL_CPU_ZIP_THREADS`, zlib workers with `compression: cpu` |
| `remote.node`, `remote.ip`, `remote.client_ip` | — | remote mode: the pool node, its RDMA NIC IP, and the GPU node's RDMA NIC IP that routes to it |
| `remote.port` | 5555 | first of the 32 host ports the model takes on both nodes: the RDMA daemon's (`IAXL_RDMA_DAEMON_PORT`, then one per rank) and, from `port + 16`, IAXL's API (`IAXL_API_CONTROLLER_PORT`, `IAXL_API_WORKER_BASE_PORT`). Give models that share a pool node separate blocks |
| `remote.cpu` | 16 × TP | daemon vCPUs |
| `daemon_overhead_gb` | 32 | daemon memory per rank beside its pools |

A remote pool also needs the model's `node:` (the GPU node that owns `client_ip`), since the vLLM pod binds that
address. `nemotron-mini-4b-kvshrink-rdma` in `models.yaml` is a remote example to copy.

Deploy as any model: `./model-manager deploy nemotron-mini-4b-kvshrink --wait`. `--dry-run` prints the
`LLMInferenceService` and, in remote mode, the pool daemon's Deployment.

## Sizing

- **Local:** the pod requests `memory + (pool_gb + scratch_gb) × TP`; `memory` is vLLM's own (category default
  64Gi). The CPU/memory pre-flight checks that a node has it free.
- **Remote:** the vLLM pod requests only `memory`; the daemon requests `(pool_gb + scratch_gb + daemon_overhead_gb) ×
  TP` and `remote.cpu`, reserved whether or not the pool has filled (IAXL grows the pool up to `pool_gb` and evicts
  from there). The pre-flight checks what is **free** on the pool node once its other pods' requests are counted
  (not the daemon it replaces): memory, CPU, `rdma/hca` and the QAT VFs and IAA work queues, and `rdma/hca` on the GPU
  node. A memory shortfall names the largest `pool_gb` that fits.
- **Host ports:** the pre-flight reads the other remote-pool models' daemons (`app.kubernetes.io/component=kv-pool`)
  and refuses a second one on the same GPU node, or a port block that collides with another on a shared node, naming
  a free `remote.port`.

Leave the OS and other pods their share: on a 512 GiB node, a 256–400 GiB pool is typical.

## Measured

B300 GPU node → Xeon 6980P pool node (8 QAT, 8 IAA) over 400G RoCE, Nemotron-Mini-4B (TP 1), IAXL `556c832` on
vLLM 0.29.0; 4 concurrent ~3,850-token prompts (2 GB of KV), offloaded then reloaded with identical output:

| Compression | Offload | Reload | Ratio |
|---|---:|---:|---:|
| `cpu` (4 threads) | 0.56 GB/s | 1.72 GB/s | 1.37× |
| `cpu`, `cpu_zip_threads: 12` | 1.48 GB/s | 4.16 GB/s | 1.37× |
| `off` | 4.73 GB/s | 8.33 GB/s | 1.00× |
| `qat+iaa` | 8.08 GB/s | 11.34 GB/s | 1.38× |

Compression buys pool capacity, not bandwidth; QAT/IAA are faster than storing raw because they move data by DMA.

## Known limits

- One remote-pool model per GPU node: KServe serves every `LLMInferenceService` on port 8000 and does not support
  another, and on the host network that port is the node's. Several models can share a pool node, each with its own
  `remote.port` block and its own daemon and pool (pools are not shared between models).
- A remote-pool model runs one replica: its vLLM pod is pinned to the node that owns `client_ip`.
- The pods take the node's whole cpuset share and pin OMP threads inside it (the image's entrypoint); `--cpu`
  sets the request.
- A burst larger than one scheduler step (`--max-num-batched-tokens`) offloaded only that step's requests with
  IAXL main on vLLM 0.29; keep benchmark bursts within one step.
- DSA + GDRCopy (`IAXL_DSA_GD_ENABLE`) is not wired: it was never validated.
- Not on OpenShift: the runtime has its own container command.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Runtime 'vllm-cuda-kvshrink' version '0.29.0' defines no image` | build and push the image, and set it (above) |
| `KV offload pre-flight: ... enforces Pod Security` | remote mode: use a namespace without `baseline`/`restricted` enforcement |
| Pool daemon `Pending` on `rdma/hca`, `qat.intel.com/dc` or `iaa.intel.com/wq-user-shared` | list that node under `accel_device_plugins_nodes` (`rdma`, `qat`, `iaa`) and re-run `install accel_device_plugins` |
| Daemon log `qat_zip: failed to start QAT user space` | the pod got no usable VF: check its `qat.intel.com/dc` request and `device_ownership_from_security_context` |
| vLLM log `UCX ERROR ... no route to <ip>` | the entrypoint logs the `UCX_IB_GID_INDEX` it chose for the RDMA IP; check `client_ip`/`ip` are on the RDMA NICs |
| First requests take seconds (`CuTeDSL JIT compilation`) | vLLM 0.29 compiles kernels per batch shape; warm up before measuring |
