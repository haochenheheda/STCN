#/bin/bash
#for i in 3 5 7 9 11 
#do
#python eval_youtube.py --output output_kmn${i}_res480 --memory_type kmn --sigma ${i}
#done

#for i in 7 9 11 
#do
#python eval_youtube.py --output output_kmn${i}_res600 --memory_type kmn --sigma ${i} --res 600
#done

#python eval_youtube.py --output output_bs32_g1 --model saves/retrain_b32_g1_s03/retrain_b32_g1_s03_100000.pth

python eval_youtube_ms.py --output output_ms_test --vname ffd7c15f47
