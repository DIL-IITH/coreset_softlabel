# python3 -W ignore train.py --dataset cifar100 --gpuid $1 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 --task-name all-data --base-dir ./data-model/cifar100
# python3 -W ignore generate_importance_score.py --gpuid $1 --base-dir ./data-model/cifar100 --task-name all-data --dataset cifar100
N_NEIGHBOR=5
GAMMA=0.1
for TEMPERATURE in 10 20 30 
do
    for CORESET_RATIO in 0.01 0.05 0.1 0.2 
    do
        python3 -W ignore train_sl.py --dataset cifar10 --gpuid $1 --iterations 40000 --task-name class-lb-graph-n=$N_NEIGHBOR-g=$GAMMA-$CORESET_RATIO --base-dir ./data-model/cifar100/class/ --coreset --coreset-mode class --budget-mode uniform --sampling-mode graph \
        --data-score-path ./data-model/cifar10/all-data/data-score-all-data.pickle \
        --coreset-key forgetting \
        --feature-path ./data-model/cifar10/all-data/train-features-all-data.npy \
        --coreset-ratio $CORESET_RATIO --mis-ratio 0.4 --label-balanced \
        --n-neighbor $N_NEIGHBOR --gamma $GAMMA --stratas 25 --graph-mode sum --graph-sampling-mode weighted --arch_path $2 --temperature $TEMPERATURE
    done 
done 

