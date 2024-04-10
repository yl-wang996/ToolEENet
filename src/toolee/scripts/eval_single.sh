export CUDA_VISIBLE_DEVICES=0
python runners/evaluation_single.py \
--score_model_dir ScoreNet/ckpt_epoch2000.pth \
--score_model_dir ScoreNet/ckpt_epoch2000.pth \
--log_folder /dataSSD/yunlong/dataspace/training_logs_obj_pose \
--data_path /dataSSD/yunlong/dataspace/DatasetToolEE \
--eval_set test \
--result_dir /data/yunlong/training_logs_obj_pose/results \
--sampler_mode ode \
--max_eval_num 1000000 \
--percentage_data_for_test 1.0 \
--batch_size 200 \
--seed 0 \
--test_source val \
--eval_repeat_num 50 \
--T0 0.55 \
# --save_video \
