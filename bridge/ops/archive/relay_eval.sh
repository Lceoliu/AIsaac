#!/bin/bash
# Relay a finished run's last.pt to the 3080 and queue its 128-seed eval there: relay_eval.sh <tag> <host-spec> <remote last.pt>
# host-spec: 3090 | 3080 (3080 = already local to the eval host, just queue)
set -e
tag=$1; host=$2; ck=$3; ev=${4:-ev128.sh}   # 4th arg: ev128_mem.sh for B20 memory checkpoints
S="C:/Users/eolc/AppData/Local/Temp/claude/D--Projects-fortune-Isaac/b7955aa6-bfa7-4d4d-bac8-9a9b795c4ef8/scratchpad/ck"
mkdir -p "$S"
dest=/home/eolc/isaac-abplus/bld/runs/ev128/ck-$tag.pt
if [ "$host" = 3090 ]; then
  scp -o BatchMode=yes rhythmo@10.19.131.132:"$ck" "$S/$tag-last.pt" > /dev/null
  scp -o BatchMode=yes -P 2222 "$S/$tag-last.pt" eolc@100.76.185.120:"$dest" > /dev/null
else
  ssh -o BatchMode=yes -p 2222 eolc@100.76.185.120 "cp '$ck' '$dest'" < /dev/null
fi
ssh -o BatchMode=yes -p 2222 eolc@100.76.185.120 "cd ~/isaac-abplus/bld && setsid nohup bash $ev $tag $dest > runs/ev128/$tag.queue.log 2>&1 < /dev/null & sleep 2; pgrep -fc '$ev $tag'" < /dev/null
