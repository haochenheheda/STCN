#/bin/bash
#for i in 3 5 7 9 11 
#do
#python eval_youtube.py --output output_kmn${i}_res480 --memory_type kmn --sigma ${i}
#done

for i in 7 9 11 
do
python eval_youtube.py --output output_kmn${i}_res600 --memory_type kmn --sigma ${i} --res 600
done
