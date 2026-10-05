#!/bin/bash
# Set the Arc Pro B70's power cap through the xe driver's hwmon (run as root). The cap resets when the driver
# reloads, so run it at boot and periodically (docs/RUNNING.md has a systemd service + timer).
#   sudo scripts/b70/power-cap.sh [WATTS]        (default 120)
# B70_ID: the card's PCI vendor:device id (lspci -nn); the first matching card is capped.
set -eu
WATTS=${1:-120}
B70_ID=${B70_ID:-8086:e223}
bdf=$(lspci -D -d "$B70_ID" | awk 'NR==1 {print $1}')
[ -n "$bdf" ] || { echo "no device $B70_ID"; exit 0; }
for cap in /sys/bus/pci/devices/$bdf/hwmon/hwmon*/power1_cap; do
  [ -f "$cap" ] || continue
  echo $((WATTS * 1000000)) > "$cap"
  echo "$bdf: power cap $(( $(cat "$cap") / 1000000 )) W"
done
