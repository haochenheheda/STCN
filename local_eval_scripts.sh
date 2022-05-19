for iteration in 280000 282000 284000 286000 288000 290000 292000 294000 296000 298000 300000
do
#python general_eval_youtube.py --output output/resnet50_topk_additional_${iteration} --model saves/retrain_b16_g1_s012_k_resnet50_additional/retrain_b16_g1_s012_k_resnet50_additional_${iteration}.pth --memory_type topk --value_encoder_type resnet18 --key_encoder_type resnet50
#python general_eval_youtube.py --output output/resnet50_aspp_topk_additional_${iteration} --model saves/retrain_b16_g1_s012_k_resnet50_aspp_additional/retrain_b16_g1_s012_k_resnet50_aspp_additional_${iteration}.pth --memory_type topk --value_encoder_type resnet18 --key_encoder_type resnet50 --aspp
#python general_eval_youtube.py --output output/wide_resnet50_topk_additional_${iteration} --model saves/retrain_b16_g1_s012_k_wide_resnet50_additional/retrain_b16_g1_s012_k_wide_resnet50_additional_${iteration}.pth --memory_type topk --value_encoder_type resnet18 --key_encoder_type wide_resnet50
python general_eval_youtube.py --output output/wide_resnet50_aspp_topk_additional_${iteration} --model saves/retrain_b16_g1_s012_k_wide_resnet50_aspp_additional/retrain_b16_g1_s012_k_wide_resnet50_aspp_additional_${iteration}.pth --memory_type topk --value_encoder_type resnet18 --key_encoder_type wide_resnet50 --aspp
done
