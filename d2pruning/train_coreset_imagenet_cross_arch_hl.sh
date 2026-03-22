N_NEIGHBOR=5
GAMMA=0.1
for CORESET_RATIO in 0.005 0.01 
do
           python3 -W ignore train_imagenet_cross_arch_hl.py --dataset imagenet --gpuid $1 --iterations 40000 --task-name class-lb-graph-n=$N_NEIGHBOR-g=$GAMMA-$CORESET_RATIO --base-dir ./data-model/imagenet/graph/ --coreset --coreset-mode class --budget-mode uniform --sampling-mode graph \
        --data-score-path ./data-model/imagenet/all-data/data-score-all-data.pickle \
        --coreset-key accumulated_margin \
        --network resnet34 \
        --target vit \
        --feature-path ./data-model/imagenet/all-data/train-features-all-data.npy \
        --coreset-ratio $CORESET_RATIO --mis-ratio 0.4 --label-balanced \
        --n-neighbor $N_NEIGHBOR --gamma $GAMMA --stratas 25 --graph-mode sum --graph-sampling-mode weighted
    done
