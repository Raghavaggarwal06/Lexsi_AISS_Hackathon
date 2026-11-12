#!/bin/bash
# Serial GPU queue for round 7 (round 6's queue): runs the lines of /content/build/queue7.txt in order, one at a time; lines may be appended
# while it runs. `touch queue7.stop` ends it once the queue is empty. The log is mirrored into runs_private/round7/logs/ after every line.
cd /content/build || exit 1
mkdir -p runs_private/round7/logs
touch queue7.txt queue7.done queue7.log
while true; do
  n=$(wc -l < queue7.done); line=$(sed -n "$((n+1))p" queue7.txt)
  if [ -z "$line" ]; then
    if [ -f queue7.stop ]; then break; fi
    sleep 5; continue
  fi
  echo "$(date -u +%H:%M:%S) START $line" >> queue7.log
  bash -c "$line"; rc=$?
  echo "$(date -u +%H:%M:%S) END($rc) $line" >> queue7.log
  echo "$line" >> queue7.done
  cp queue7.log runs_private/round7/logs/queue7.log
  python -c "import sys; sys.path.insert(0, 'scripts'); import r7lib; r7lib.sync()" 2>/dev/null
done
