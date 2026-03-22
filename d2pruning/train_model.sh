N_NEIGHBOR=5
GAMMA=0.1
CORESET_RATIO=$1
python3 -W ignore train.py --dataset cifar10 --gpuid $2 --iterations 40000 --task-name class-lb-graph-n=$N_NEIGHBOR-g=$GAMMA-$CORESET_RATIO \
    --base-dir ./data-model/cifar10/class/ --coreset --coreset-mode class --budget-mode uniform --sampling-mode graph \
    --data-score-path ./data-model/cifar10/all-data/data-score-all-data.pickle \
    --feature-path ./data-model/cifar10/all-data/train-features-all-data.npy \
    --coreset-key forgetting --coreset-ratio $CORESET_RATIO --mis-ratio 0.4 --label-balanced \
    --n-neighbor $N_NEIGHBOR --gamma $GAMMA --stratas 25 --graph-mode sum --graph-sampling-mode weighted