#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-38423}
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

echo "Distributed: $MASTER_ADDR:$PORT"
echo "rank: $NODE_RANK/$NNODES"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
