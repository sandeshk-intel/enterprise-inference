"""KV-cache offload (KVShrink / IAXL): the model's DDR pool, in its own pod or in a pool daemon on another node that the
vLLM pod reaches over RDMA. See docs/customize/kv-offload.md."""
from __future__ import annotations

import ipaddress
import json
import math

from . import k8s
from .config import KvOffload, Model, is_uint, pick, text
from .policy import _cpu_millis, gate, mem_bytes, pod_requests
from .renderer import _list, quote, render, resource_lines, template
from .ui import abort, info, warn

COMPRESSION = ("cpu", "off", "qat", "iaa", "qat+iaa")
HOST_NETWORK_FORBIDDEN = ("baseline", "restricted")
GIB = 2 ** 30
# A remote pool takes PORT_BLOCK host ports from remote.port on its GPU node and its pool node: the RDMA daemon's
# (port, then one per rank) and, from port + API_OFFSET, IAXL's management API (controller, then one per rank).
PORT_BLOCK, API_OFFSET = 32, 16
# KServe serves every LLMInferenceService on 8000 and does not support another port; on the host network, that
# is one remote-pool model per GPU node.
VLLM_PORT = 8000
ANNOTATION = "model-manager/kv-pool"


def pool_name(m: Model) -> str:
    return f"{m.name}-kv-pool"


def _uint(c: dict, key: str, m: Model, default: object = "", where: str = "kv_offload") -> int:
    value = text(pick(c, key, default))
    if not (is_uint(value) and int(value) > 0):
        abort(f"Model '{m.name}': {where}.{key} must be a positive integer, got: {value or 'nothing'}")
    return int(value)


def _ipv4(remote: dict, key: str, m: Model) -> str:
    value = text(pick(remote, key))
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        abort(f"Model '{m.name}': kv_offload.remote.{key} must be the IPv4 address of an RDMA NIC, got: "
              f"{value or 'nothing'}")
    return value


def resolve(m: Model, tp: int) -> KvOffload | None:
    """The model's kv_offload over its runtime's (plumbing in runtimes/<name>/values.yaml); None without one."""
    plumbing = pick(m.rt, "kv_offload", {})
    if not plumbing:
        if m.kv_offload:
            abort(f"Model '{m.name}' sets kv_offload, which needs a KV-offload runtime such as vllm-cuda-kvshrink, "
                  f"not '{m.runtime}'.")
        return None
    c = {**plumbing, **m.kv_offload}
    compression = text(pick(c, "compression", "cpu"))
    if compression not in COMPRESSION:
        abort(f"Model '{m.name}': kv_offload.compression must be one of {', '.join(COMPRESSION)}, got: {compression}")
    qat, iaa = "qat" in compression, "iaa" in compression
    kv = KvOffload(pool_gb=_uint(c, "pool_gb", m), scratch_gb=_uint(c, "scratch_gb", m), compression=compression,
                   pool_env={k: text(v) for k, v in pick(c, "pool_env", {}).items()}, resources=pick(c, "resources", {}),
                   qat_vfs=_uint(c, "qat_vfs", m) if qat else 0, iaa_wqs=_uint(c, "iaa_wqs", m) if iaa else 0,
                   qat_devices=text(pick(c, "qat_devices")))
    for key in ("zip_threads", "cpu_zip_threads"):
        if pick(c, key):
            setattr(kv, key, str(_uint(c, key, m)))
    if qat and tp > 1 and not kv.qat_devices:
        abort(f"Model '{m.name}': QAT with tensor parallel size {tp} needs kv_offload.qat_devices, the QAT devices of "
              "each rank '|'-separated (e.g. \"0,1,2,3|4,5,6,7\"): which VFs a rank gets is not predictable.")
    remote = pick(c, "remote", {})
    if not remote:
        return kv
    if not isinstance(remote, dict) or not text(pick(remote, "node")):
        abort(f"Model '{m.name}': kv_offload.remote needs node: (the node that runs the pool daemon), ip: and client_ip:.")
    kv.pool_node = text(remote["node"])
    if not m.node:
        abort(f"Model '{m.name}': a remote KV pool needs node: (the GPU node whose NIC owns kv_offload.remote.client_ip).")
    if len(pool_name(m)) > 63:
        abort(f"Model '{m.name}': the name is too long for its pool daemon '{pool_name(m)}' (63 characters at most).")
    kv.pool_ip, kv.client_ip = _ipv4(remote, "ip", m), _ipv4(remote, "client_ip", m)
    kv.port = str(_uint(remote, "port", m, pick(c, "rdma_port"), "kv_offload.remote"))
    if not 1024 <= int(kv.port) <= 65536 - PORT_BLOCK or VLLM_PORT in range(int(kv.port), int(kv.port) + PORT_BLOCK):
        abort(f"Model '{m.name}': kv_offload.remote.port must be between 1024 and {65536 - PORT_BLOCK}, and the "
              f"{PORT_BLOCK} ports from it must leave out vLLM's {VLLM_PORT}.")
    if tp >= API_OFFSET:
        abort(f"Model '{m.name}': a remote KV pool supports a tensor parallel size below {API_OFFSET}, got {tp}.")
    kv.pool_cpu = str(_uint(remote, "cpu", m, 16 * tp, "kv_offload.remote"))
    kv.overhead_gb = _uint(c, "daemon_overhead_gb", m)
    kv.pool_memory_gi = (kv.pool_gb + kv.scratch_gb + kv.overhead_gb) * tp
    return kv


def _api_env(kv: KvOffload) -> dict:
    return {"IAXL_API_CONTROLLER_PORT": str(int(kv.port) + API_OFFSET),
            "IAXL_API_WORKER_BASE_PORT": str(int(kv.port) + API_OFFSET + 1)}


def _accelerators(kv: KvOffload, tp: int) -> dict:
    """QAT VFs (every rank's) and IAA work queues of the pod that compresses."""
    return {**({text(kv.resources["qat"]): str(kv.qat_vfs * tp)} if kv.qat_vfs else {}),
            **({text(kv.resources["iaa"]): str(kv.iaa_wqs)} if kv.iaa_wqs else {})}


def _pool_env(kv: KvOffload, cpus: int) -> dict:
    """Env of the pod that holds the pool and compresses: block format, pool sizes and the compression backend. The
    enable flags are always set: IAXL turns QAT on by default and aborts when the pod has no QAT VF."""
    qat, iaa = bool(kv.qat_vfs), bool(kv.iaa_wqs)
    env = {**kv.pool_env, "IAXL_DDR_POOL_SIZE_GB": str(kv.pool_gb), "IAXL_SCRATCH_POOL_SIZE_GB": str(kv.scratch_gb),
           "IAXL_KV_COMPRESSION": "0" if kv.compression == "off" else "1",
           "IAXL_CPU_ZIP_ENABLE": "0" if qat or iaa else "1",
           "IAXL_QAT_ZIP_ENABLE": "1" if qat else "0", "IAXL_IAA_ZIP_ENABLE": "1" if iaa else "0"}
    # IAXL's default, one thread per accelerator worker, oversubscribes the pod.
    threads = kv.zip_threads or (str(max(cpus - 4, 1)) if (qat or iaa) and cpus else "")
    if threads:
        env["IAXL_OMP_THREAD_NUM"] = threads
    if kv.cpu_zip_threads and not (qat or iaa):
        env["IAXL_CPU_ZIP_THREADS"] = kv.cpu_zip_threads
    if qat:
        devices = kv.qat_devices or ",".join(str(d) for d in range(kv.qat_vfs))
        env.update(QAT_POLICY=str(kv.qat_vfs), KVSHRINK_QAT_DEVICES=devices,
                   IAXL_QAT_INSTANCE_NUM=str(len(devices.split("|")[0].split(",")) * 4))
    if iaa:
        env.update(IAXL_IAA_DEVICES="auto", IAXL_IAA_INSTANCE_NUM=str(kv.iaa_wqs * 4))
    return env


def apply(m: Model, kv: KvOffload, tp: int) -> None:
    """The vLLM pod's side. Local: the pool in its env, memory and devices. Remote: an RDMA client of the pool daemon,
    on the host network because UCX binds the NIC that owns client_ip."""
    if kv.remote:
        env = {"IAXL_RDMA_ENABLE": "1", "IAXL_RDMA_DAEMON_NIC_IPS": kv.pool_ip, "IAXL_RDMA_CLIENT_NIC_IPS": kv.client_ip,
               "KVSHRINK_RDMA_LOCAL_IP": kv.client_ip, "IAXL_RDMA_DAEMON_PORT": kv.port, "IAXL_RDMA_TP_SIZE": str(tp),
               **_api_env(kv)}
        m.extra_resources = {text(kv.resources["rdma"]): "1"}
        m.host_network = True
        info(f"  KV offload: {kv.pool_gb} GiB x {tp} rank(s) on {kv.pool_node} ({kv.pool_ip}:{kv.port}) over RDMA, "
             f"compression {kv.compression}")
    else:
        env = _pool_env(kv, int(m.cpu) if is_uint(m.cpu) else 0)
        m.extra_resources = _accelerators(kv, tp)
        m.memory = f"{math.ceil(mem_bytes(m.memory) / GIB) + (kv.pool_gb + kv.scratch_gb) * tp}Gi"
        info(f"  KV offload: {kv.pool_gb} GiB x {tp} rank(s) in the pod (memory {m.memory}), compression {kv.compression}")
    m.env = {**env, **m.env}
    m.kv = kv


def pool_manifest(m: Model, tp: int) -> str:
    """The remote pool daemon: the model's image running iaxl.remote_pool.daemon on kv.pool_node's host network."""
    kv = m.kv
    env = {**pick(m.rt, "env", {}), **_pool_env(kv, int(kv.pool_cpu)), "IAXL_RDMA_ENABLE": "1",
           "KVSHRINK_RDMA_LOCAL_IP": kv.pool_ip, "KVSHRINK_EXEC": "python3", **_api_env(kv)}
    args = ["-X", "faulthandler", "-m", "iaxl.remote_pool.daemon", "--nic-ips", kv.pool_ip, "--port", kv.port,
            "--tp-size", str(tp)]
    return render(template("kv-pool.yaml"), NAME=pool_name(m), MODEL_NAME=m.name, NAMESPACE=m.namespace,
                  IMAGE=m.image, NODE=quote(kv.pool_node), CPU=kv.pool_cpu, MEMORY=f"{kv.pool_memory_gi}Gi",
                  ARGS=_list([f"            - {quote(a)}" for a in args]),
                  ENV=_list([f"            - name: {k}\n              value: {quote(text(v))}" for k, v in sorted(env.items())]),
                  RESOURCES=resource_lines({text(kv.resources["rdma"]): "1", **_accelerators(kv, tp)}, 14),
                  RUN_AS=m.run_as_user,
                  FOOTPRINT=quote(json.dumps({"client_node": m.node, "pool_node": kv.pool_node, "port": int(kv.port)})))


def _quantity(key: str, value: object) -> int:
    """cpu in millis, memory in bytes, an extended resource as a count; 0 when unparseable."""
    value = text(value)
    return _cpu_millis(value) if key == "cpu" else mem_bytes(value) if key == "memory" else \
        int(value) if is_uint(value) else 0


def _free(node: str, needs: dict, owner: str) -> tuple[dict, dict] | None:
    """Per key of `needs` (cpu in millis, memory in bytes, extended resources as counts): the node's allocatable and
    what its pods request, not counting `owner`'s own pods (being replaced); None when the node or pods can't be read."""
    n = k8s.get_json("node", node)
    pods = k8s.get_json("pods", "-A", "--field-selector",
                        f"spec.nodeName={node},status.phase!=Succeeded,status.phase!=Failed")
    if not n or pods is None:
        return None
    alloc = pick(pick(n, "status", {}), "allocatable", {})
    have = {k: _quantity(k, pick(alloc, k, "0")) for k in needs}
    used = dict.fromkeys(needs, 0)
    for p in pods.get("items", []):
        if pick(pick(pick(p, "metadata", {}), "labels", {}), "app.kubernetes.io/name") == owner:
            continue
        cpu, memory = pod_requests(p)
        for k in needs:
            used[k] += cpu if k == "cpu" else memory if k == "memory" else sum(
                _quantity(k, pick(pick(pick(c, "resources", {}), "requests", {}), k, "0"))
                for c in pick(pick(p, "spec", {}), "containers", []))
    return have, used


def _short(node: str, needs: dict, owner: str, kv: KvOffload | None = None, tp: int = 1) -> list[str]:
    """What `node` lacks of `needs` once its other pods' requests are counted. With kv (the pool node), a memory
    shortfall names the largest pool_gb that fits."""
    state = _free(node, needs, owner)
    if state is None:
        return [f"Node {node} was not found, or its pods cannot be read."]
    have, used = state
    errors = []
    for key, need in needs.items():
        free = have[key] - used[key]
        if free >= need:
            continue
        if key == "memory" and kv:
            fit = free // GIB // tp - kv.scratch_gb - kv.overhead_gb
            errors.append(f"Node {node} has {max(free, 0) // GIB}Gi memory free ({have[key] // GIB}Gi allocatable, "
                          f"{used[key] // GIB}Gi requested by its other pods); the KV pool needs {need // GIB}Gi = "
                          f"(pool_gb {kv.pool_gb} + scratch {kv.scratch_gb} + overhead {kv.overhead_gb}) x TP {tp}. "
                          + (f"Lower pool_gb to {fit} or less." if fit > 0 else "Not even the smallest pool fits."))
        elif key == "cpu":
            errors.append(f"Node {node} has {max(free, 0) / 1000:g} CPU free of {have[key] / 1000:g}; the KV pool "
                          f"needs {need / 1000:g}.")
        else:
            errors.append(f"Node {node} has {max(free, 0)} {key} free of {have[key]}; the KV pool needs {need}.")
    return errors


def _footprint(client_node: str, pool_node: str, port: int) -> dict:
    """Host ports a remote-pool model holds, per node: its port block on both, and vLLM's on the GPU node."""
    block = set(range(port, port + PORT_BLOCK))
    return {pool_node: set(block), client_node: block | {VLLM_PORT}}


def _port_conflicts(m: Model) -> list[str]:
    """Other remote-pool models whose host ports collide with this one's on a node they share."""
    pools = k8s.get_json("deployments", "-A", "-l", "app.kubernetes.io/component=kv-pool")
    if pools is None:
        warn("Could not list the KV pool daemons — skipping the host-port check.")
        return []
    kv = m.kv
    others = []
    for d in pools.get("items", []):
        meta = pick(d, "metadata", {})
        if (pick(meta, "name"), pick(meta, "namespace")) == (pool_name(m), m.namespace):
            continue
        try:
            fp = json.loads(text(pick(pick(meta, "annotations", {}), ANNOTATION)))
            others.append((f"{pick(meta, 'namespace')}/{pick(meta, 'name')}", fp["client_node"],
                           _footprint(fp["client_node"], fp["pool_node"], int(fp["port"]))))
        except (ValueError, KeyError, TypeError):
            continue
    mine = _footprint(m.node, kv.pool_node, int(kv.port))
    taken: dict = {}
    for _, _, theirs in others:
        for node, ports in theirs.items():
            taken.setdefault(node, set()).update(ports)
    errors = []
    for name, client_node, theirs in others:
        if client_node == m.node:
            errors.append(f"{name}'s model already runs vLLM on {m.node}'s host network, on port {VLLM_PORT}, which "
                          "KServe fixes: one remote-pool model per GPU node. Use another GPU node, or a local pool.")
        elif any(ports & mine.get(node, set()) for node, ports in theirs.items()):
            errors.append(f"Host ports {kv.port}-{int(kv.port) + PORT_BLOCK - 1} collide with {name}'s on a node they "
                          f"share. Set kv_offload.remote.port to {_free_port(taken, mine)}.")
    return errors


def _free_port(taken: dict, mine: dict) -> int:
    """The first port block from 5555 that is clear, on each of this model's nodes, of the ports taken there."""
    port = 5555
    while any(set(range(port, port + PORT_BLOCK)) & (taken.get(node, set()) | {VLLM_PORT}) for node in mine):
        port += PORT_BLOCK
    return port


def preflight(m: Model, tp: int) -> list[str]:
    """What would leave its pods Pending, crashing or rejected: the devices, CPU and memory free on the nodes, the
    host ports other remote-pool models hold, and, for the host network, the namespace's Pod Security level."""
    kv = m.kv
    accel = {k: int(v) for k, v in _accelerators(kv, tp).items()}
    if not kv.remote:
        return _short(m.node, accel, m.name) if m.node and accel else []
    labels = pick(pick(k8s.get_json("namespace", m.namespace), "metadata", {}), "labels", {})
    level = text(pick(labels, "pod-security.kubernetes.io/enforce"))
    errors = [f"Namespace {m.namespace} enforces Pod Security '{level}', which rejects the host network the RDMA pool "
              "needs: deploy the model to a namespace without it."] if level in HOST_NETWORK_FORBIDDEN else []
    rdma = text(kv.resources["rdma"])
    pool_needs = {rdma: 1, **accel, "cpu": int(kv.pool_cpu) * 1000, "memory": kv.pool_memory_gi * GIB}
    return (errors + _port_conflicts(m) + _short(m.node, {rdma: 1}, m.name)
            + _short(kv.pool_node, pool_needs, pool_name(m), kv, tp))


def check(m: Model, tp: int) -> None:
    gate("KV offload", m.name, lambda: preflight(m, tp))


def reconcile(m: Model, tp: int) -> None:
    """Apply the remote pool daemon and wait for it before the vLLM pod starts; a local pool leaves none behind."""
    if not m.kv.remote:
        return k8s.delete("deployment", pool_name(m), m.namespace, quiet=True)
    k8s.apply(pool_manifest(m, tp), quiet=True)
    info(f"Waiting for the KV pool daemon {pool_name(m)} on {m.kv.pool_node}...")
    r = k8s.run("rollout", "status", f"deployment/{pool_name(m)}", "-n", m.namespace, "--timeout=600s")
    if r.returncode:
        abort(f"KV pool daemon {pool_name(m)} is not ready: {r.stderr.strip() or r.stdout.strip()}. Check: kubectl "
              f"describe pod -n {m.namespace} -l app.kubernetes.io/name={pool_name(m)}")
