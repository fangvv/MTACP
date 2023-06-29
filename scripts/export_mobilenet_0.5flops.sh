python amc_search.py \
    --job=export \
    --model=mobilenet \
    --dataset=imagenet \
    --data_root=/dataset/imagenet \
    --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
    --seed=2018 \
    --n_calibration_batches=300 \
    --n_worker=32 \
    --channels=3,24,48,96,80,192,200,328,352,368,360,328,400,736,752 \
    --export_path=./checkpoints/mobilenet_0.5flops_export.pth.tar


python amc_search.py \
    --job=export \
    --model=mobilenet \
    --dataset=cifar10 \
    --data_root=./dataset/cifar10 \
    --ckpt_path=./checkpoints/mobilenetamc_ckpt.pth \
    --seed=2021 \
    --n_calibration_batches=300 \
    --n_worker=32 \
    --channels=3,24,64,104,64,152,96,104,200,104,376,104,104,968,656 \
    --export_path=./checkpoints/mobilenet_cifar10_0.5flops_export.pth.tar
