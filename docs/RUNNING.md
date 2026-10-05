# Running

Assumes the engine is built and the model prepared ([`BUILDING.md`](BUILDING.md)). One B70, 32 GB host RAM.

## Start the server

```sh
M=~/models/qwen38-flash-next
export QWFN_B70_LLAMA=~/src/llama-b70
export QWFN_B70_GSQ=$M/gsq-rco/Q2_0
export QWFN_B70_HEAD=$M/overlay/v4/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00008.gguf
export QWFN_B70_MTP=$M/mtp/mtp-q2-00001-of-00003.gguf
export QWFN_B70_DRAFT_VOCAB=$M/mtp/draft_vocab_lat60.bin
export QWFN_RAM_EXCLUSIVE=1 NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0
sudo prlimit --memlock=unlimited --pid $$        # once per shell: the RAM tier is locked memory
scripts/b70/qwfn-b70.sh --mmproj $M/gsq-rco/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --vram 23 --ram 11 --prefix-cache 5 --timeshare 3 --quantum 30 --linger 1 --host 127.0.0.1 --port 8080
```

`qwfn-b70.sh` sets all the kernel and engine switches; anything you append overrides its flags (the last value of a
repeated flag wins). Every flag and variable: [`PARAMETERS.md`](PARAMETERS.md).

- Wait for `curl localhost:8080/health` to answer. The **first request after a new build is slow** (GPU kernels compile); later
  starts reuse the compiler cache in `$HOME`, so run with a writable home.
- The log should say `locked (VmLck ...)`. If not, the memlock limit was not raised.
- Out of memory on the GPU at start or with several sessions: lower `--vram` by 1.

Test it (OpenAI-compatible; `/v1/messages` speaks Anthropic's API):

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":64}'
```

Under systemd, use `LimitMEMLOCK=infinity` and `Environment=` lines for the exports above, and
`ExecStart=/path/to/QwFNfer-xpu/scripts/b70/qwfn-b70.sh ...`.

## The knobs worth tuning

| Flag | Above | What it trades |
|---|---|---|
| `--vram GB` | 23 | The VRAM expert tier. More = fewer experts read per token = faster decode. 23 leaves room for the MTP head, vision and 3 sessions; 24 is fine for one session at a time. GPU OOM -> go down. |
| `--ram GB` | 11 | The RAM expert tier (the next ~11 GB of experts; the rest comes off the NVMe). On a 32 GB host, 11 is the sweet spot: higher pushes the host into swap and everything slows down. Lower it if other programs need the RAM. |
| `--timeshare N` + `--prefix-cache GB` | 3, 5 | Up to N concurrent sessions take turns (`--quantum` s each), so a short request is not stuck behind a 100K-token prompt; the prefix cache keeps the last conversations' state in RAM so a follow-up turn skips its prefill. One user, one session: `--timeshare 1` and give the RAM to `--ram`. |

## Power cap

The card runs fine capped: 120 W loses ~11% decode and ~13% prefill against 160 W (it never draws more than
~160 W on this workload). Set the cap as root:

```sh
sudo scripts/b70/power-cap.sh 120        # watts; B70_ID=8086:xxxx if lspci shows another device id
```

The cap resets when the driver reloads, so re-apply it on a timer:

```ini
# /etc/systemd/system/b70-power-cap.service
[Unit]
Description=B70 power cap
[Service]
Type=oneshot
ExecStart=/path/to/QwFNfer-xpu/scripts/b70/power-cap.sh 120

# /etc/systemd/system/b70-power-cap.timer
[Unit]
Description=Re-apply the B70 power cap
[Timer]
OnBootSec=10s
OnUnitActiveSec=10min
[Install]
WantedBy=timers.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now b70-power-cap.timer`

## Not recommended now

In the tree, measured, and slower than the defaults on this card. Left for experiments:

| Setting | Why not |
|---|---|
| `--mtp-drafts 2` or `3` | More draft tokens per step. Each extra verified position costs ~10 ms (mostly its expert reads), more than the accepted drafts give back: 2 drafts measured 2-7% slower than 1. |
| `QWFN_MTP_MIN_P2=F` | With `--mtp-drafts >= 2`: draft the 2nd/3rd token only while the head's probability is at least F. Cuts the losses above, does not turn them into a gain. |
| `QWFN_MTP_SHARED_CHAIN=0` | Chained drafts (2nd/3rd) attend over the trunk's selected cells by default; `0` makes them attend densely (slower at long context). Only matters with `--mtp-drafts >= 2`. |
