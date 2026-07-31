import re
import sys
from collections import defaultdict

pattern = re.compile(r"RESULT_PARTIAL seq_len~(\d+) shard=(\d+)/(\d+): correct=(\d+) total=(\d+) peak=([\d.]+)")
oom_pattern = re.compile(r"RESULT_PARTIAL seq_len~(\d+) shard=(\d+)/(\d+): OOM")

agg = defaultdict(lambda: {"correct": 0, "total": 0, "peak": 0.0, "oom_shards": set()})

for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                length, shard, num_shards, correct, total, peak = m.groups()
                length = int(length)
                agg[length]["correct"] += int(correct)
                agg[length]["total"] += int(total)
                agg[length]["peak"] = max(agg[length]["peak"], float(peak))
                continue
            m2 = oom_pattern.search(line)
            if m2:
                length, shard, num_shards = m2.groups()
                agg[int(length)]["oom_shards"].add(int(shard))

print(f"{'seq_len':>10} {'accuracy':>10} {'correct/total':>15} {'peak_GiB':>10}")
for length in sorted(agg.keys(), reverse=True):
    d = agg[length]
    if d["oom_shards"]:
        print(f"{length:>10} {'OOM':>10} {'-':>15} {'-':>10}  (shards: {sorted(d['oom_shards'])})")
        continue
    if d["total"] == 0:
        continue
    acc = d["correct"] / d["total"] * 100
    print(f"{length:>10} {acc:>9.1f}% {d['correct']:>6}/{d['total']:<8} {d['peak']:>10.2f}")
