#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Resolves CPU pinning from the container's own cgroup cpuset, then execs `vllm serve "$@"` (the remote
# pool daemon sets KVSHRINK_EXEC=python3). Upstream's setvars.sh reads the host-wide NUMA cpulist, which
# fails with EINVAL when the pod's cpuset is narrower.
#   VLLM_CPU_OMP_THREADS_BIND=auto|unset  the cpuset split evenly across --tensor-parallel-size ranks, '|'-separated
#   KVSHRINK_NUMA_NODE=auto|unset         the NUMA node holding most of those CPUs
#   KVSHRINK_RDMA_LOCAL_IP=<ip>           UCX_IB_GID_INDEX = the RoCE v2 GID of that IP. With several IPs on one NIC,
#                                         UCX otherwise advertises the lowest-index GID, which the peer may not route to.
# Test hooks: KVSHRINK_CPUSET_FILE, KVSHRINK_NODE_DIR, KVSHRINK_SYSFS, KVSHRINK_NET_ADDR_FILE
set -euo pipefail

CPUSET_FILE="${KVSHRINK_CPUSET_FILE:-/sys/fs/cgroup/cpuset.cpus.effective}"
NODE_DIR="${KVSHRINK_NODE_DIR:-/sys/devices/system/node}"
# What to exec after the environment is resolved (default: the vLLM server).
EXEC_CMD="${KVSHRINK_EXEC:-vllm serve}"

# Tensor-parallel size from the arguments the chart renders (default 1).
tp=1
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --tensor-parallel-size|-tp|--tp-size) tp="${args[i + 1]:-1}" ;;
        --tensor-parallel-size=*|--tp-size=*) tp="${args[i]#*=}" ;;
    esac
done
[[ "$tp" =~ ^[1-9][0-9]*$ ]] || { echo "kvshrink-entrypoint: bad tensor-parallel size '$tp'" >&2; exit 1; }

bind="${VLLM_CPU_OMP_THREADS_BIND:-auto}"
numa="${KVSHRINK_NUMA_NODE:-auto}"

if [[ "$bind" == auto || "$numa" == auto ]]; then
    cpus=""
    [[ -r "$CPUSET_FILE" ]] && cpus="$(tr -d '[:space:]' <"$CPUSET_FILE")"
    [[ -n "$cpus" ]] || cpus="$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"
    [[ -n "$cpus" ]] || { echo "kvshrink-entrypoint: cannot read the pod's cpuset" >&2; exit 1; }

    resolved="$(python3 - "$cpus" "$tp" "$NODE_DIR" <<'PYEOF'
import collections, glob, os, sys

spec, tp, node_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def parse(s):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out

def fmt(ids):
    out, start, prev = [], ids[0], ids[0]
    for c in ids[1:]:
        if c == prev + 1:
            prev = c
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)

ids = sorted(parse(spec))
if len(ids) < tp:
    sys.exit(f"cpuset '{spec}' has {len(ids)} CPU(s), fewer than tensor-parallel size {tp}")
base, rem = divmod(len(ids), tp)
ranks, start = [], 0
for r in range(tp):
    n = base + (1 if r < rem else 0)
    ranks.append(fmt(ids[start:start + n]))
    start += n

counts = collections.Counter()
want = set(ids)
for path in glob.glob(os.path.join(node_dir, "node[0-9]*")):
    try:
        node = os.path.basename(path)[4:]
        with open(os.path.join(path, "cpulist")) as f:
            counts[node] += len(parse(f.read().strip()) & want)
    except OSError:
        pass
numa = counts.most_common(1)[0][0] if counts else "0"
print("|".join(ranks))
print(numa)
PYEOF
)"
    resolved_bind="$(sed -n 1p <<<"$resolved")"
    resolved_numa="$(sed -n 2p <<<"$resolved")"
    [[ "$bind" == auto ]] && bind="$resolved_bind"
    [[ "$numa" == auto ]] && numa="$resolved_numa"
fi

if [[ -n "${KVSHRINK_RDMA_LOCAL_IP:-}" && -z "${UCX_IB_GID_INDEX:-}" ]]; then
    gid="$(python3 - "$KVSHRINK_RDMA_LOCAL_IP" <<'PYEOF' || true
import glob, json, os, subprocess, sys
ip = sys.argv[1]
root = os.environ.get("KVSHRINK_SYSFS", "/sys")
if os.environ.get("KVSHRINK_NET_ADDR_FILE"):
    addr = json.load(open(os.environ["KVSHRINK_NET_ADDR_FILE"]))
else:
    addr = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr"]))
nic = next((l["ifname"] for l in addr for a in l.get("addr_info", []) if a.get("local") == ip), None)
if not nic:
    sys.exit(1)
want = "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" % tuple(int(x) for x in ip.split("."))
for dev in glob.glob(f"{root}/class/net/{nic}/device/infiniband/*"):
    ib = os.path.basename(dev)
    gids = glob.glob(f"{root}/class/infiniband/{ib}/ports/1/gids/*")
    for g in sorted(gids, key=lambda p: int(os.path.basename(p))):
        i = os.path.basename(g)
        if open(g).read().strip() == want:
            kind = open(f"{root}/class/infiniband/{ib}/ports/1/gid_attrs/types/{i}").read().strip()
            if kind == "RoCE v2":
                print(i)
                sys.exit(0)
sys.exit(1)
PYEOF
)"
    if [[ -n "$gid" ]]; then
        export UCX_IB_GID_INDEX="$gid"
        echo "kvshrink-entrypoint: RDMA local IP ${KVSHRINK_RDMA_LOCAL_IP} -> UCX_IB_GID_INDEX=${gid}" >&2
    else
        echo "kvshrink-entrypoint: WARNING: no RoCE v2 GID found for ${KVSHRINK_RDMA_LOCAL_IP}; UCX will choose one" >&2
    fi
fi

export VLLM_CPU_OMP_THREADS_BIND="$bind"
export KVSHRINK_NUMA_NODE="$numa"
echo "kvshrink-entrypoint: tp=${tp} VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND} KVSHRINK_NUMA_NODE=${KVSHRINK_NUMA_NODE}" >&2

# shellcheck disable=SC2086  # EXEC_CMD is intentionally word-split ("vllm serve", "python3")
exec $EXEC_CMD "$@"
