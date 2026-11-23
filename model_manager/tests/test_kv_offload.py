"""KV-cache offload (runtime vllm-cuda-kvshrink, core/kv_offload.py): a render golden, unit tests and the image's
entrypoint wrapper.

The golden (tests/golden/kv-offload-rdma-dry-run.txt) is a snapshot of the current output. Regenerate it with
`python tests/test_kv_offload.py` after an intended change.
"""
from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, k8s, kv_offload, renderer  # noqa: E402
from core.config import Catalog  # noqa: E402
from test_parity import GOLDEN, MM, ROOMY, run  # noqa: E402

LOCAL, RDMA = "nemotron-mini-4b-kvshrink", "nemotron-mini-4b-kvshrink-rdma"
IMAGE = "registry.example.com/vllm-kvshrink:v0.29.0-556c832"
ENTRYPOINT = MM / "runtimes" / "vllm-cuda-kvshrink" / "docker" / "kvshrink-entrypoint.sh"


def _catalog(tmp: Path, **overrides) -> Path:
    """models.yaml with the runtime's image set (it ships empty) and RDMA's kv_offload overridden."""
    data = yaml.safe_load((MM / "models.yaml").read_text())
    data["runtimes"]["vllm-cuda-kvshrink"]["versions"]["0.29.0"]["image"] = IMAGE
    for m in data["models"]:
        if m["name"] == RDMA:
            m["kv_offload"].update(overrides)
    path = tmp / "models.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _golden_case(tmp: Path) -> dict:
    return {"args": ("deploy", RDMA, "--dry-run"), "env": {}, "k8s": list(ROOMY), "catalog": str(_catalog(tmp)),
            "deviations": ()}


def _deploy(monkeypatch, tmp_path, name=LOCAL, **overrides):
    """The resolved model after kv_offload.apply, its manifest and, remote, the pool daemon."""
    monkeypatch.setenv("MM_CONFIG", str(_catalog(tmp_path, **overrides)))
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    catalog = Catalog.load()
    o = cli.Options()
    m = config.resolve(catalog, name, o)
    args, tp = cli._engine_cli_args(o, int(m.gpus), renderer.engine_args(m))
    kv = kv_offload.resolve(m, tp)
    kv_offload.apply(m, kv, tp)
    doc = yaml.safe_load(renderer.manifest(m, catalog, o, {}, m.cpu, args))
    pool = yaml.safe_load(kv_offload.pool_manifest(m, tp)) if kv.remote else None
    return m, doc, pool


def _env(container: dict) -> dict:
    return {e["name"]: e["value"] for e in container["env"]}


def test_render_golden(tmp_path):
    assert run(MM / "model-manager", _golden_case(tmp_path), tmp_path) == \
        (GOLDEN / "kv-offload-rdma-dry-run.txt").read_text()


def test_the_runtime_ships_without_an_image(monkeypatch):
    monkeypatch.setenv("MM_CONFIG", str(MM / "models.yaml"))
    with pytest.raises(SystemExit):
        config.resolve(Catalog.load(), LOCAL, cli.Options())


def test_local_pool_is_in_the_pod_env_and_memory(monkeypatch, tmp_path):
    m, doc, _ = _deploy(monkeypatch, tmp_path)
    c = doc["spec"]["template"]["containers"][0]
    env = _env(c)
    assert c["command"] == ["/usr/local/bin/kvshrink-entrypoint.sh"] and c["image"] == IMAGE
    assert c["args"][:5] == ["/mnt/models", "--served-model-name", LOCAL, "--port", "8000"]
    assert "--no-enable-prefix-caching" in c["args"] and "KVShrinkConnector" in c["args"][c["args"].index("--kv-transfer-config") + 1]
    assert (env["IAXL_DDR_POOL_SIZE_GB"], env["IAXL_SCRATCH_POOL_SIZE_GB"]) == ("256", "8")
    assert (env["IAXL_CPU_ZIP_ENABLE"], env["IAXL_QAT_ZIP_ENABLE"], env["IAXL_KV_COMPRESSION"]) == ("1", "0", "1")
    assert env["IAXL_RDMA_ENABLE"] == "0" and env["IAXL_ZIP_EXPONENT_ONLY"] == "1"
    assert c["resources"]["requests"]["memory"] == c["resources"]["limits"]["memory"] == "328Gi"  # 64 + 256 + 8
    assert c["resources"]["limits"] == {"memory": "328Gi", "nvidia.com/gpu": "1"}
    assert "hostNetwork" not in doc["spec"]["template"]
    assert c["securityContext"]["capabilities"] == {"drop": ["ALL"]} and c["securityContext"]["runAsNonRoot"]


def test_model_env_still_wins_over_the_pool_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MM_CONFIG", str(_catalog(tmp_path)))
    m = config.resolve(Catalog.load(), LOCAL, cli.Options())
    m.env = {"IAXL_SCRATCH_POOL_SIZE_GB": "4"}
    kv_offload.apply(m, kv_offload.resolve(m, 1), 1)
    assert m.env["IAXL_SCRATCH_POOL_SIZE_GB"] == "4" and m.env["IAXL_DDR_POOL_SIZE_GB"] == "256"


def test_remote_client_is_an_rdma_client_on_the_host_network(monkeypatch, tmp_path):
    m, doc, _ = _deploy(monkeypatch, tmp_path, RDMA)
    spec = doc["spec"]["template"]
    c = spec["containers"][0]
    env = _env(c)
    assert spec["hostNetwork"] is True and spec["dnsPolicy"] == "ClusterFirstWithHostNet"
    assert (env["IAXL_RDMA_ENABLE"], env["IAXL_RDMA_DAEMON_NIC_IPS"], env["IAXL_RDMA_CLIENT_NIC_IPS"]) == \
        ("1", "192.0.2.13", "192.0.2.14")
    assert env["KVSHRINK_RDMA_LOCAL_IP"] == "192.0.2.14" and env["IAXL_RDMA_TP_SIZE"] == "1"
    assert "IAXL_DDR_POOL_SIZE_GB" not in env and env["IAXL_QAT_ZIP_ENABLE"] == "0"
    assert c["resources"]["limits"] == {"memory": "64Gi", "nvidia.com/gpu": "1", "rdma/hca": "1"}
    assert "gpu-node-1" in json.dumps(spec["affinity"])


def test_remote_pool_daemon_holds_the_pool_and_compresses_with_qat_and_iaa(monkeypatch, tmp_path):
    _, _, pool = _deploy(monkeypatch, tmp_path, RDMA)
    spec = pool["spec"]["template"]["spec"]
    c = spec["containers"][0]
    env = _env(c)
    assert pool["metadata"]["name"] == f"{RDMA}-kv-pool" and spec["hostNetwork"] is True
    assert spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0][
        "matchExpressions"][0]["values"] == ["pool-node-1"]
    assert c["image"] == IMAGE and c["args"][3:] == ["iaxl.remote_pool.daemon", "--nic-ips", "192.0.2.13", "--port",
                                                     "5555", "--tp-size", "1"]
    assert c["resources"]["limits"] == {"cpu": "16", "memory": "440Gi", "rdma/hca": "1", "qat.intel.com/dc": "8",
                                        "iaa.intel.com/wq-user-shared": "8"}  # (400 + 8 + 32) x 1
    assert (env["IAXL_DDR_POOL_SIZE_GB"], env["KVSHRINK_EXEC"], env["KVSHRINK_RDMA_LOCAL_IP"]) == ("400", "python3", "192.0.2.13")
    assert (env["IAXL_CPU_ZIP_ENABLE"], env["IAXL_QAT_ZIP_ENABLE"], env["IAXL_IAA_ZIP_ENABLE"]) == ("0", "1", "1")
    assert (env["QAT_POLICY"], env["KVSHRINK_QAT_DEVICES"], env["IAXL_QAT_INSTANCE_NUM"]) == ("8", "0,1,2,3,4,5,6,7", "32")
    assert (env["IAXL_IAA_INSTANCE_NUM"], env["IAXL_OMP_THREAD_NUM"]) == ("32", "12")  # vCPUs - 4
    assert c["securityContext"]["capabilities"] == {"drop": ["ALL"]} and c["securityContext"]["runAsUser"] == 1000


@pytest.mark.parametrize("compression, cpu_zip, kv_compression, qat", [
    ("off", "1", "0", "0"), ("qat", "0", "1", "1"), ("iaa", "0", "1", "0")])
def test_compression_backends(monkeypatch, tmp_path, compression, cpu_zip, kv_compression, qat):
    _, _, pool = _deploy(monkeypatch, tmp_path, RDMA, compression=compression)
    env = _env(pool["spec"]["template"]["spec"]["containers"][0])
    assert (env["IAXL_CPU_ZIP_ENABLE"], env["IAXL_KV_COMPRESSION"], env["IAXL_QAT_ZIP_ENABLE"]) == (cpu_zip, kv_compression, qat)


@pytest.mark.parametrize("override, message", [
    ({"pool_gb": 0}, "pool_gb must be a positive integer"),
    ({"compression": "zstd"}, "compression must be one of"),
    ({"remote": {"node": "pool-node-1", "ip": "pool-node-1", "client_ip": "192.0.2.14"}}, "remote.ip must be the IPv4"),
    ({"remote": {"ip": "192.0.2.13", "client_ip": "192.0.2.14"}}, "remote needs node:"),
])
def test_invalid_kv_offload_aborts(monkeypatch, tmp_path, capsys, override, message):
    with pytest.raises(SystemExit):
        _deploy(monkeypatch, tmp_path, RDMA, **override)
    assert message in capsys.readouterr().err


def test_qat_with_tensor_parallelism_needs_explicit_devices(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MM_CONFIG", str(_catalog(tmp_path)))
    m = config.resolve(Catalog.load(), RDMA, cli.Options())
    with pytest.raises(SystemExit):
        kv_offload.resolve(m, 2)
    assert "needs kv_offload.qat_devices" in capsys.readouterr().err
    m.kv_offload["qat_devices"] = "0,1,2,3|4,5,6,7"
    kv = kv_offload.resolve(m, 2)
    assert kv.pool_memory_gi == (400 + 8 + 32) * 2 and kv.pool_cpu == "32"


def test_remote_pool_needs_the_gpu_node_and_a_kv_runtime_needs_a_pool(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MM_CONFIG", str(_catalog(tmp_path)))
    m = config.resolve(Catalog.load(), RDMA, cli.Options())
    m.node = ""
    with pytest.raises(SystemExit):
        kv_offload.resolve(m, 1)
    m.kv_offload = {}
    with pytest.raises(SystemExit):
        kv_offload.resolve(m, 1)
    err = capsys.readouterr().err
    assert "needs node:" in err and "pool_gb must be a positive integer" in err


def test_kv_offload_on_another_runtime_aborts(monkeypatch, capsys):
    monkeypatch.setenv("MM_CONFIG", str(MM / "models.yaml"))
    m = config.resolve(Catalog.load(), "qwen3-0-6b-cuda", cli.Options())
    assert kv_offload.resolve(m, 1) is None
    m.kv_offload = {"pool_gb": 10}
    with pytest.raises(SystemExit):
        kv_offload.resolve(m, 1)
    assert "needs a KV-offload runtime" in capsys.readouterr().err


def _nodes(monkeypatch, nodes: dict, enforce: str = ""):
    def get_json(kind, name=None, *rest):
        if kind == "namespace":
            return {"metadata": {"labels": {"pod-security.kubernetes.io/enforce": enforce} if enforce else {}}}
        return {"status": {"allocatable": nodes[name]}} if name in nodes else None
    monkeypatch.setattr(k8s, "get_json", get_json)


def test_preflight_passes_when_the_nodes_have_the_devices_and_memory(monkeypatch, tmp_path):
    m, _, _ = _deploy(monkeypatch, tmp_path, RDMA)
    _nodes(monkeypatch, {"gpu-node-1": {"rdma/hca": "63"},
                         "pool-node-1": {"rdma/hca": "63", "qat.intel.com/dc": "128", "iaa.intel.com/wq-user-shared": "8",
                                         "memory": "1000Gi"}})
    assert kv_offload.preflight(m, 1) == []


def test_preflight_reports_missing_devices_memory_and_a_forbidding_namespace(monkeypatch, tmp_path):
    m, _, _ = _deploy(monkeypatch, tmp_path, RDMA)
    _nodes(monkeypatch, {"pool-node-1": {"rdma/hca": "63", "qat.intel.com/dc": "4", "memory": "256Gi"}},
           enforce="restricted")
    errors = "\n".join(kv_offload.preflight(m, 1))
    assert "enforces Pod Security 'restricted'" in errors and "Node gpu-node-1 was not found" in errors
    assert "4 allocatable qat.intel.com/dc; the KV pool needs 8" in errors
    assert "0 allocatable iaa.intel.com/wq-user-shared" in errors and "256Gi allocatable memory; the KV pool needs 440Gi" in errors


def test_reconcile_applies_and_waits_for_the_daemon_and_a_local_pool_removes_it(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(k8s, "apply", lambda manifest, **kw: calls.append(("apply", yaml.safe_load(manifest)["kind"])))
    monkeypatch.setattr(k8s, "delete", lambda *a, **kw: calls.append(("delete",) + a[:2]))
    monkeypatch.setattr(k8s, "run", lambda *a, **kw: calls.append(a[:3]) or subprocess.CompletedProcess(a, 0, "", ""))
    m, _, _ = _deploy(monkeypatch, tmp_path, RDMA)
    kv_offload.reconcile(m, 1)
    local, _, _ = _deploy(monkeypatch, tmp_path)
    kv_offload.reconcile(local, 1)
    assert calls == [("apply", "Deployment"), ("rollout", "status", f"deployment/{RDMA}-kv-pool"),
                     ("delete", "deployment", f"{LOCAL}-kv-pool")]


def test_undeploy_removes_the_pool_daemon_of_a_kv_model_only(monkeypatch, tmp_path):
    deleted = []
    monkeypatch.setattr(k8s, "delete", lambda kind, name, *a, **kw: deleted.append((kind, name)))
    for name in (RDMA, "qwen3-0-6b-cuda"):
        monkeypatch.setenv("MM_CONFIG", str(_catalog(tmp_path)))
        m = config.resolve(Catalog.load(), name, cli.Options())
        monkeypatch.setattr(cli.routing, "deregister", lambda m: None)
        monkeypatch.setattr(cli.routing, "litellm_deregister", lambda name: None)
        cli._remove(m, wait=False, timeout=0)
    assert ("deployment", f"{RDMA}-kv-pool") in deleted
    assert not any(kind == "deployment" for kind, name in deleted if "qwen" in name)


# ── The image's entrypoint wrapper ───────────────────────────────────────────


@pytest.fixture
def host(tmp_path):
    """A fake host: node0 = CPUs 0-3, node1 = 4-7, and one RoCE NIC with two IPs, so two RoCE v2 GIDs
    (index 3 = 192.168.0.8, index 5 = 10.201.1.13)."""
    for node, cpus in (("node0", "0-3"), ("node1", "4-7")):
        (tmp_path / "node" / node).mkdir(parents=True)
        (tmp_path / "node" / node / "cpulist").write_text(cpus)
    vllm = tmp_path / "fake-vllm"
    vllm.write_text('#!/usr/bin/env bash\necho "BIND=$VLLM_CPU_OMP_THREADS_BIND NUMA=$KVSHRINK_NUMA_NODE '
                    'GID=${UCX_IB_GID_INDEX:-unset} ARGS=$*"\n')
    vllm.chmod(0o755)
    port = tmp_path / "sys" / "class" / "infiniband" / "mlx5_0" / "ports" / "1"
    (tmp_path / "sys" / "class" / "net" / "eth9" / "device" / "infiniband" / "mlx5_0").mkdir(parents=True)
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "types").mkdir(parents=True)
    for i, (gid, kind) in enumerate([("fe80:0000:0000:0000:966d:aeff:fe4e:5308", "IB/RoCE v1"),
                                     ("fe80:0000:0000:0000:966d:aeff:fe4e:5308", "RoCE v2"),
                                     ("0000:0000:0000:0000:0000:ffff:c0a8:0008", "IB/RoCE v1"),
                                     ("0000:0000:0000:0000:0000:ffff:c0a8:0008", "RoCE v2"),
                                     ("0000:0000:0000:0000:0000:ffff:0ac9:010d", "IB/RoCE v1"),
                                     ("0000:0000:0000:0000:0000:ffff:0ac9:010d", "RoCE v2")]):
        (port / "gids" / str(i)).write_text(gid)
        (port / "gid_attrs" / "types" / str(i)).write_text(kind)
    (tmp_path / "addr.json").write_text(json.dumps([
        {"ifname": "eth9", "addr_info": [{"local": "192.168.0.8"}, {"local": "10.201.1.13"}]},
        {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]}]))
    return tmp_path


def _entry(host: Path, cpuset: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    (host / "cpuset").write_text(cpuset)
    base = {"PATH": os.environ["PATH"], "KVSHRINK_CPUSET_FILE": str(host / "cpuset"),
            "KVSHRINK_NODE_DIR": str(host / "node"), "KVSHRINK_EXEC": str(host / "fake-vllm"),
            "KVSHRINK_SYSFS": str(host / "sys"), "KVSHRINK_NET_ADDR_FILE": str(host / "addr.json")}
    return subprocess.run(["bash", str(ENTRYPOINT), *args], env={**base, **env},  # nosec B603 B607: test fixture
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("cpuset, args, env, expected", [
    ("0-7", ["--tensor-parallel-size", "2", "--model", "x"], {},
     "BIND=0-3|4-7 NUMA=0 GID=unset ARGS=--tensor-parallel-size 2 --model x"),
    ("2,3,5-7", [], {}, "BIND=2-3,5-7 NUMA=1 GID=unset ARGS="),                      # node1 holds 3 of 5
    ("0-4", ["--tensor-parallel-size=2"], {}, "BIND=0-2|3-4 NUMA=0 GID=unset ARGS=--tensor-parallel-size=2"),
    ("0-7", [], {"VLLM_CPU_OMP_THREADS_BIND": "0-1", "KVSHRINK_NUMA_NODE": "1"}, "BIND=0-1 NUMA=1 GID=unset ARGS="),
    ("0-7", ["--tp-size", "2"], {}, "BIND=0-3|4-7 NUMA=0 GID=unset ARGS=--tp-size 2"),  # the pool daemon's flag
    ("0-7", [], {"KVSHRINK_RDMA_LOCAL_IP": "10.201.1.13"}, "BIND=0-7 NUMA=0 GID=5 ARGS="),  # not the lowest (3)
    ("0-7", [], {"KVSHRINK_RDMA_LOCAL_IP": "192.168.0.8"}, "BIND=0-7 NUMA=0 GID=3 ARGS="),
    ("0-7", [], {"KVSHRINK_RDMA_LOCAL_IP": "10.201.1.13", "UCX_IB_GID_INDEX": "1"}, "BIND=0-7 NUMA=0 GID=1 ARGS="),
    ("0-7", [], {"KVSHRINK_RDMA_LOCAL_IP": "10.9.9.9"}, "BIND=0-7 NUMA=0 GID=unset ARGS="),
])
def test_entrypoint_pins_cpus_from_the_pods_cpuset_and_picks_the_roce_gid(host, cpuset, args, env, expected):
    r = _entry(host, cpuset, *args, **env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


def test_entrypoint_fails_fast_with_fewer_cpus_than_ranks(host):
    assert _entry(host, "0", "--tensor-parallel-size", "2").returncode != 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        (GOLDEN / "kv-offload-rdma-dry-run.txt").write_text(run(MM / "model-manager", _golden_case(Path(d)), Path(d)))
