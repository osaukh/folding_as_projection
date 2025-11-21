| weight_decay | save_dir                                  | warmup_steps | max_lr |
| ------------ | ----------------------------------------- | ------------ | ------ |
| 0.01         | checkpoints/llama_60m-2025-10-04-13-24-22 | 880          | 0.001  |
                sparsity    mag     fold    
                    0       32.11
                    0.2     54.51   47.17
                    0.5     398.62  221.32
| 0.01         | checkpoints/llama_60m-2025-10-04-13-02-41 | 1100         | 0.001  |
                sparsity    mag     fold    
                    0       32.14
                    0.2     50.11   46.75
                    0.5     220.54  172.57
| 0.01         | checkpoints/llama_60m-2025-10-04-13-21-53 | 2200         | 0.001  |
                sparsity    mag     fold    
                    0       32.20
                    0.2     46.57   47.54
                    0.5     174.58  216.36
| 0            | checkpoints/llama_60m-2025-10-04-12-58-51 | 880          | 0.001  |
                sparsity    mag     fold    
                    0       32.17
                    0.2     51.14   48.23
                    0.5     220.33  223.86
| 0            | checkpoints/llama_60m-2025-10-04-12-57-40 | 1100         | 0.001  |
                sparsity    mag     fold    
                    0       32.21
                    0.2     50.03   47.47
                    0.5     231.41  204.47
| 0            | checkpoints/llama_60m-2025-10-04-12-58-12 | 2200         | 0.001  |
                sparsity    mag     fold    
                    0       32.40
                    0.2     46.38   46.92
                    0.5     177.48  185.27
| 0.01         | checkpoints/llama_60m-2025-10-04-15-16-16 | 880          | 0.005  |
                sparsity    mag     fold    
                    0       30.12
                    0.2     68.70   55.32              
                    0.5     641.69  302.43      
| 0.01         | checkpoints/llama_60m-2025-10-04-15-15-20 | 1100         | 0.005  |
                sparsity    mag     fold    
                    0       29.77
                    0.2     68.29   49.81
                    0.5     564.96  234.56
| 0.01         | checkpoints/llama_60m-2025-10-04-15-17-15 | 2200         | 0.005  |
                sparsity    mag     fold    
                    0       29.60
                    0.2     54.50   47.04  
                    0.5     360.52  208.02
| 0            | checkpoints/llama_60m-2025-10-04-14-34-38 | 880          | 0.005  |
                sparsity    mag     fold    
                    0       30.47
                    0.2     78.73   62.35
                    0.5     762.05  395.04
| 0            | checkpoints/llama_60m-2025-10-04-13-26-16 | 1100         | 0.005  |
                sparsity    mag     fold    
                    0       30.17
                    0.2     59.20   49.58
                    0.5     544.87  184.74
| 0            | checkpoints/llama_60m-2025-10-04-13-31-04 | 2200         | 0.005  |
                sparsity    mag     fold    
                    0       29.75
                    0.2     56.18   46.55       
                    0.5     353.35  165.21
| 0.01         | checkpoints/llama_60m-2025-10-04-11-05-37 | 2200         | 0.01   |
                sparsity    mag     fold    
                    0       29.25
                    0.2     51.46   44.28                     
                    0.5     323.68  288.83                      
| 0.01         | checkpoints/llama_60m-2025-10-04-11-04-07 | 880          | 0.01   |
                sparsity    mag     fold    
                    0       31.82
                    0.2     66.98   51.80
                    0.5     910.48  406.75
| 0.01         | checkpoints/llama_60m-2025-10-04-10-41-56 | 1100         | 0.01   |
                sparsity    mag     fold    
                    0       29.85
                    0.2     102.41  67.69
                    0.5     977.92  367.94
| 0            | checkpoints/llama_60m-2025-10-04-10-40-24 | 2200         | 0.01   |
                sparsity    mag     fold    
                    0       29.57
                    0.2     54.43   47.77  
                    0.5     351.11  209.06
| 0            | checkpoints/llama_60m-2025-10-04-10-41-40 | 880          | 0.01   |
                sparsity    mag     fold    
                    0       108.56
                    0.2     129.77  123.85     
                    0.5     279.17  198.72
| 0            | checkpoints/llama_60m-2025-10-04-10-42-13 | 1100         | 0.01   |
                sparsity    mag     fold    
                    0       30.31
                    0.2     97.97   61.19
                    0.5     860.14  533.62


python3 main.py --model_name_or_path ../checkpoints/llama/checkpoints/llama_60m-2025-10-04-12-58-12/model_11001 --model meta-llama/Llama-2-7b-hf --model_config configs/llama_60m.json --cache_dir ../data/llms --device cuda:4 --dtype float16 --pruning_ratio 0.2 --nsamples 128 --prune_method mag_sp

python3 main.py --model_name_or_path ../checkpoints/llama/checkpoints/llama_60m-2025-10-04-13-24-22/model_11001 --model meta-llama/Llama-2-7b-hf --model_config configs/llama_60m.json --cache_dir ../data/llms --device cuda:5 --dtype float16 --pruning_ratio 0.2 --nsamples 128 --prune_method folding

