python amc_search.py \
    --job=train \
    --model=mobilenet \
    --dataset=imagenet \
    --preserve_ratio=0.5 \
    --lbound=0.2 \
    --rbound=1 \
    --reward=acc_reward \
    --data_root=./dataset/imagenet \
    --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
    --seed=2018

python amc_search.py \
    --job=train \
    --model=resnet50 \
    --dataset=cifar10 \
    --preserve_ratio=0.5 \
    --lbound=0.2 \
    --rbound=1 \
    --reward=acc_reward \
    --data_root=./dataset/cifar10/ \
    --ckpt_path=./checkpoints/resnet50_cifar10.pth.tar \
    --seed=2021

