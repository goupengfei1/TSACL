export CUDA_VISIBLE_DEVICES=$1

nitr=5


python -u run.py --TSACon --TSACon_wnorm ReVIN   --TSACon_lambda 1.0  --decoder_desize 96  --d_model 2 --d_ff 12 --e_layers 3 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id TSACon --model TemporalCon --data ETTh2 --seq_len 96  --label_len 0 --pred_len 96   --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.001  --train_epochs 10 --feature S
python -u run.py --TSACon --TSACon_wnorm LastVal --TSACon_lambda 1.0  --decoder_desize 336 --d_model 8 --d_ff 32 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id TSACon --model TemporalCon --data ETTh2 --seq_len 336 --label_len 0 --pred_len 720  --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.002  --train_epochs 10 --feature S
python -u run.py --TSACon --TSACon_wnorm LastVal --TSACon_lambda 1.0  --decoder_desize 168 --d_model 8 --d_ff 24 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id TSACon --model TemporalCon --data ETTh2 --seq_len 168 --label_len 0 --pred_len 1440 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.002  --train_epochs 10 --feature S
python -u run.py --TSACon --TSACon_wnorm LastVal --TSACon_lambda 0.05 --decoder_desize 168 --d_model 8 --d_ff 24 --e_layers 2 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id TSACon --model TemporalCon --data ETTh2 --seq_len 168 --label_len 0 --pred_len 2160 --enc_in 1 --des 'Exp' --itr $nitr --batch_size 64 --learning_rate 0.0005 --train_epochs 10 --feature S
