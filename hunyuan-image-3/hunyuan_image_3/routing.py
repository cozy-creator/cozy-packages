"""The expert-hit record of a stage (se-079): which routed experts each forward used and with how many tokens,
and what an expert cache of a given size would have held: least recently used, and the most any cache of that
size could (next-use eviction, the minimum bytes to move)."""

from __future__ import annotations

import heapq
import statistics
from array import array
from collections import OrderedDict

#: Expert caches replayed, in experts: 256 is ~9.7 GB at int8, 1,024 ~38.7 GB.
CACHES = (256, 512, 1024)


class Routing:
    """Called as `trace(layer, counts)` by every MoE layer of every forward, in dispatch order."""

    def __init__(self, experts: int = 64, caches: tuple[int, ...] = CACHES) -> None:
        self.experts, self.caches = experts, caches
        self.forwards = 0
        self.hit_per_layer = array("I")
        self.tokens_per_hit = array("I")
        #: `layer * experts + expert` of every expert a forward ran, in the order it ran them.
        self.sequence = array("I")

    def __call__(self, layer: int, counts: list[int]) -> None:
        self.forwards += layer == 0
        hit = 0
        for expert, tokens in enumerate(counts):
            if tokens:
                hit += 1
                self.tokens_per_hit.append(tokens)
                self.sequence.append(layer * self.experts + expert)
        self.hit_per_layer.append(hit)

    def summary(self) -> dict[str, float]:
        """Per layer and forward means; `working_set` counts the distinct experts of the stage; per cache size the
        hit rate of LRU and of next-use eviction, and the latter's misses per forward (experts that must move)."""
        if not self.sequence:
            return {}
        found: dict[str, float] = {
            "forwards": self.forwards,
            "experts_hit_per_layer_mean": round(statistics.fmean(self.hit_per_layer), 2),
            "experts_hit_per_layer_min": min(self.hit_per_layer),
            "tokens_per_hit_expert_min": min(self.tokens_per_hit),
            "tokens_per_hit_expert_median": statistics.median(self.tokens_per_hit),
            "tokens_per_hit_expert_max": max(self.tokens_per_hit),
            "working_set": len(set(self.sequence)),
        }
        for size in self.caches:
            found[f"lru_{size}_hit_rate"] = round(lru_hits(self.sequence, size) / len(self.sequence), 4)
            hits = min_hits(self.sequence, size)
            found[f"min_{size}_hit_rate"] = round(hits / len(self.sequence), 4)
            found[f"min_{size}_misses_per_forward"] = round((len(self.sequence) - hits) / max(self.forwards, 1), 1)
        return found


def lru_hits(sequence: array[int] | list[int], size: int) -> int:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for key in sequence:
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            cache[key] = None
            if len(cache) > size:
                cache.popitem(last=False)
    return hits


def min_hits(sequence: array[int] | list[int], size: int) -> int:
    """Belady's MIN: on a miss with the cache full, evict the expert used furthest in the future."""
    never = len(sequence)
    upcoming = [never] * len(sequence)
    seen: dict[int, int] = {}
    for i in range(len(sequence) - 1, -1, -1):
        upcoming[i] = seen.get(sequence[i], never)
        seen[sequence[i]] = i
    cached: dict[int, int] = {}  # key -> its next use
    furthest: list[tuple[int, int]] = []  # (-next use, key); stale entries are skipped
    hits = 0
    for i, key in enumerate(sequence):
        if key in cached:
            hits += 1
        elif len(cached) >= size:
            while True:
                use, victim = heapq.heappop(furthest)
                if cached.get(victim) == -use:
                    del cached[victim]
                    break
        cached[key] = upcoming[i]
        heapq.heappush(furthest, (-upcoming[i], key))
    return hits
