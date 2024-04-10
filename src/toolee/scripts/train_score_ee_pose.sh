export CUDA_VISIBLE_DEVICES=1
python runners/trainer.py \
--data_path /dataSSD/yunlong/dataspace/DatasetToolEE \
--log_folder /dataSSD/yunlong/dataspace/training_logs_ee_pose \
--log_dir ScoreNet \
--agent_type score \
--sampler_mode ode \
--batch_size 600 \
--sampling_steps 500 \
--eval_freq 1 \
--n_epochs 2000 \
--percentage_data_for_train 1.0 \
--percentage_data_for_test 1.0 \
--seed 0 \
--is_train \
--task_type ee_pose \
