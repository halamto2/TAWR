conda activate TAWR

L1_LOSS=1
MASK_LOSS=1
GAN_LOSS=5e-4

LEARNING_RATE=1e-3
DISC_LEARNING_RATE=1e-3

INPUT_SIZE=256
DATASET=clwd

GPU_ID=1

NAME="train_tawr"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u scripts/train.py \
 --epochs 100 \
 --schedule 65 \
 --lr ${LEARNING_RATE} \
 --dlr ${DISC_LEARNING_RATE} \
 --gpu_id ${GPU_ID} \
 --checkpoint /path/to/tensorboard/checkpoint \
 --dataset_dir /dataset/path \
 --lambda_l1 ${L1_LOSS} \
 --lambda_mask ${MASK_LOSS} \
 --lambda_gan ${GAN_LOSS} \
 --input-size ${INPUT_SIZE} \
 --train-batch 8 \
 --test-batch 1 \
 --name ${NAME} \
 --dataset ${DATASET} \
 --workers 16