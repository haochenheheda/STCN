import os
import sys
import numpy as np


result1 = sys.argv[1]
result2 = sys.argv[2]

compare_results = open('compare_results.txt','w')

good_bad_thresthod = 0.8
margin = 0.1

result_dic1 = {}
result_dic2 = {}

##################################################################################
with open(result1,'r') as f:
    lines = f.readlines()
    for line in lines:
        video_id,ob_id,_,J,F = line.strip().split(',')
        video_ob_id = '{}_{}'.format(video_id, ob_id)
        J, F = float(J), float(F)

        if video_ob_id not in result_dic1.keys():
            result_dic1[video_ob_id] = []
        result_dic1[video_ob_id].append((J + F)/2)

compare_results.write('********************bad for result1********************\n')
for k,v in result_dic1.items():
    v = np.array(v).mean()
    if v < good_bad_thresthod:
        compare_results.write(k + ' ' + str(v) + '\n')
    result_dic1[k] = np.array(v).mean()
##################################################################################
with open(result2,'r') as f:
    lines = f.readlines()
    for line in lines:
        video_id,ob_id,_,J,F = line.strip().split(',')
        video_ob_id = '{}_{}'.format(video_id, ob_id)
        J, F = float(J), float(F)

        if video_ob_id not in result_dic2.keys():
            result_dic2[video_ob_id] = []
        result_dic2[video_ob_id].append((J + F)/2)

compare_results.write('********************bad for result2********************\n')
for k,v in result_dic2.items():
    v = np.array(v).mean()
    if v < good_bad_thresthod:
        compare_results.write(k + ' ' + str(v) + '\n')
    result_dic2[k] = np.array(v).mean()
##################################################################################
compare_results.write('******************result1 >> result2******************\n')
for k in result_dic1.keys():
    if result_dic1[k] - result_dic2[k] > margin:
        compare_results.write(k + ' ' + str(result_dic1[k]) + ' ' + str(result_dic2[k]) + '\n')
##################################################################################
compare_results.write('******************result1 << result2******************\n')
for k in result_dic1.keys():
    if result_dic2[k] - result_dic1[k] > margin:
        compare_results.write(k + ' ' + str(result_dic1[k]) + ' ' + str(result_dic2[k]) + '\n')





