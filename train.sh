CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  8 \
--database SJTU  \
--data_dir_2d path_to_sjtu_projections/ \
--data_dir_pc path_to_sjtu_patch_2048/ \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 9 \
>> logs/sjtu_mmpcqa.log


CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  8 \
--database WPC  \
--data_dir_2d path_to_wpc_projections/ \
--data_dir_pc path_to_wpc_patch_2048/ \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 5 \
>> logs/wpc_mmpcqa.log

CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  8 \
--database WPC2.0  \
--data_dir_2d path_to_wpc2.0_projections/ \
--data_dir_pc path_to_wpc2.0_patch_2048/ \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 4 \
>> logs/wpc2.0_mmpcqa.log



