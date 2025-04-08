#!/usr/bin/env bash
if [[ -z "$GPU" ]]; then echo "GPU is not defined" && exit 1; fi
if [[ -z "$ROBOT" ]]; then echo "ROBOT is not defined" && exit 1; fi
if [[ -z "$IMG" ]]; then echo "IMG is not defined" && exit 1; fi
if [[ -z "$CMD" ]]; then echo "CMD is not defined" && exit 1; fi
if [[ -z "$TASK" ]]; then echo "TASK is not defined, setting to empty"; export TASK=''; fi

export DISPLAY=:0.${GPU}

echo -e "\t IMG: ${IMG}"
echo -e "\t ROBOT: ${ROBOT}"
echo -e "\t GPU: ${GPU}"
echo -e "\t TASK: ${TASK}"
echo -e "\t CMD: ${CMD}"

rndstring=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 5 ; echo '')
name="${ROBOT}_${TASK}_${SUFFIX}_${rndstring}"
wandb docker-run --rm -d --pull always --shm-size=120gb --runtime runc --name=${name} \
  --gpus="device=$GPU" \
  -e ROBOT=$ROBOT \
  -e CMD="${CMD}" \
  -e TASK=$TASK \
  dhonerkamp/mobile_rl:${IMG} bash ros_startup_incl_train.sh || exit
  sleep 1
  nohup docker logs -f "$(docker ps -ql)" > "logs_${SUFFIX}_$(date +%F)_${ROBOT}_${TASK}_$(uname -n)_$(date +"%T").txt" 2>&1 &

