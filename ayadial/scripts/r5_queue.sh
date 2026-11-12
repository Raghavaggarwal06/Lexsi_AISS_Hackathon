#!/bin/bash
# Serial GPU queue for round 5: runs the lines of /content/build/queue5.txt one at a time, in order.
# Lines can be appended while it runs. `touch queue5.stop` ends it once the queue is empty.
cd /content/build
touch queue5.txt queue5.done queue5.log
while true; do
  n_done=$(wc -l < queue5.done)
  line=$(sed -n "$((n_done + 1))p" queue5.txt)
  if [ -z "$line" ]; then
    if [ -f queue5.stop ]; then break; fi
    sleep 5
    continue
  fi
  echo "$(date -u +%H:%M:%S) START $line" >> queue5.log
  bash -c "$line"
  rc=$?
  echo "$(date -u +%H:%M:%S) END($rc) $line" >> queue5.log
  echo "$line" >> queue5.done
  cp queue5.log runs_private/round5/queue5.log 2>/dev/null
done
