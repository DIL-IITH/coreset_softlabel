#!/bin/bash 
for fraction in 0.005 0.01 0.05 0.1 0.2
do
    python3 -W ignore main_hard_label.py --dataset CIFAR100 --num_exp 5 --optimizer SGD --submodular GraphCut -se 10 --lr 0.1 --batch 128 --model ResNet18 -sp ./result --fraction $fraction --data_path datasets --selection $1 --gpu $2 --device $3 --test_interval 5 --eval_start_epoch 150 
done 


