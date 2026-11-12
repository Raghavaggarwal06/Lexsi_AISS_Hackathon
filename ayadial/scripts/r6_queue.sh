#!/bin/bash
# Serial GPU queue for round 6: runs the lines of queue6.txt in order, one at a time; new lines may be appended while it runs.
cd /content/build || exit 1
touch queue6.txt queue6.done
while true; do
  n=$(wc -l < queue6.done); line=$(sed -n "$((n+1))p" queue6.txt)
  if [ -z "$line" ]; then sleep 5; continue; fi
  echo "$(date -u +%H:%M:%S) START $line" >> queue6.log
  bash -c "$line"; rc=$?
  echo "$(date -u +%H:%M:%S) END($rc) $line" >> queue6.log
  echo "$line" >> queue6.done
done
