export CUDA_VISIBLE_DEVICES=0
python runners/trainer.py \
--data_path /dataSSD/yunlong/dataspace/DatasetToolEE \
--log_folder /dataSSD/yunlong/dataspace/training_logs_ee_pose_symtr \
--percentage_data_for_val 0.1 \
--batch_size 200 \
--eval_batch_size 200 \
--log_dir ScoreNet \
--agent_type score \
--sampler_mode ode \
--sampling_steps 500 \
--eval_freq 1 \
--n_epochs 10000 \
--percentage_data_for_train 1.0 \
--percentage_data_for_test 1.0 \
--seed 0 \
--is_train \
--task_type ee_pose \
--regression_head Rx_Ry_and_T_and_Symtr \
--pose_mode rot_matrix_symtr \
