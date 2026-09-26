# The B70 configuration

Details of `scripts/b70/qwfn-b70.sh` that the README does not carry: the engine defaults, why the flags have their
values, and the benchmark tools. The configurations (preferred: overlay v3; validated: overlay v2), their switches
and measured effects are in the README; the overlays
are built by `scripts/b70/build-overlay.sh` ([`dense-overlay.md`](dense-overlay.md)); what was tried and rejected
is [`B70-registry.md`](B70-registry.md).

## Switches

Every switch in `scripts/b70/qwfn-b70.sh` is a row of the README's **Major improvements** table (what it changes and
its measured effect); the README's run configuration lists them by group.

Engine defaults in this fork (no switch): the late fold sized to the promotion budget (`QWFN_LATE_FULL=1`
restores), deferred speculative prefetch (`QWFN_PREFETCH_EARLY=1` restores).

## Flags

The flag set is in the README's run configuration. Why these values:

- `--ctx 131072` is the per-session cap; the engine runs one sequence at a time.
- `--vram 24` leaves the ~2 GB this host needs free after a long prompt; `--ram 8` plus the 3 GB prefix
  cache plus ~4 GB for the system is what a 31 GB machine can give.
- `--prefix-cache 3`: host-memory checkpoints of conversations switched away from (~2 at 64K, one at 130K).

## Benchmarks

`tools/perf`: `decab.py` (warm decode with the `/stats` decode split), `ctxdec.py` (prefill and decode after
a long prompt from a fixture: `{"messages": [...]}`), `mtpprobe.py` (tok/s, draft acceptance and step cost),
`pcswitch.py` (two alternating conversations: the prefix cache at work), and the `mv_bench` / `moe_bench`
kernel benches (CMake targets: `mv_bench BACKEND_DIR`).
