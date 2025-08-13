@echo off
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py ^
--model RETFound_mae ^
--finetune RETFound_mae_natureCFP ^
--task RETFound_mae_natureCFP-Papil ^
--data_path F:\retfound_papil_finetune\RETFound_MAE\data ^
--input_size 224 ^
--nb_classes 3 ^
--batch_size 16 ^
--global_pool ^
--savemodel ^
--epochs 100 ^
--blr 5e-3 ^
--layer_decay 0.65 ^
--weight_decay 0.05 ^
--drop_path 0.2 ^
--world_size 1
