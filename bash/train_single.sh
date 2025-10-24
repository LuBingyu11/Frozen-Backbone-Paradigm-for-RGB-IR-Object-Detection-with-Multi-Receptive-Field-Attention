modality=$1

python train_single.py \
Datasets/M3FD \
--dataset=m3fd_${modality} \
--model=efficientdetv2_dt \
--batch-size=8 \
--amp \
--lr=1e-3 \
--opt adam \
--sched plateau \
--num-classes=6 \
--save-images \
--workers=8 \
--pretrained \
--mean 0.53584253 0.53584253 0.53584253 \
--std 0.24790472 0.24790472 0.24790472