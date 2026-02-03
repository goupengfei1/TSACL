export CUDA_VISIBLE_DEVICES=0

nitr=5


python -u run.py --TSACon --TSACon_wnorm Decomp --decomp_size 2  --decoder_desize 336 --TSACon_lambda 2.0 --d_model 8 --d_ff 128 --e_layers 1 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id TSACon --model TemporalCon --data electricity --seq_len 336 --label_len 0 --pred_len 96   --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0005 --feature S
python -u run.py --TSACon --TSACon_wnorm Decomp --decomp_size 6  --decoder_desize 336 --TSACon_lambda 2.0 --d_model 8 --d_ff 128 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id TSACon --model TemporalCon --data electricity --seq_len 336 --label_len 0 --pred_len 720  --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0005 --feature S
python -u run.py --TSACon --TSACon_wnorm Decomp --decomp_size 12 --decoder_desize 336 --TSACon_lambda 1.0 --d_model 8 --d_ff 128 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id TSACon --model TemporalCon --data electricity --seq_len 336 --label_len 0 --pred_len 1440 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0005 --feature S
python -u run.py --TSACon --TSACon_wnorm Decomp --decomp_size 6  --decoder_desize 168 --TSACon_lambda 1.0 --d_model 8 --d_ff 64  --e_layers 2 --target OT --c_out 1 --root_path ./dataset/electricity --data_path electricity.csv --model_id TSACon --model TemporalCon --data electricity --seq_len 168 --label_len 0 --pred_len 2160 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0005 --feature S
