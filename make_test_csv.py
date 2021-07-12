import pandas as pd
import numpy as np
import os
from glob import glob

def sort_fun(file):
    file_list = file.split('/')
    file_names_int = int(file_list[-1].split('.')[0])
    return (file_list[-3], file_list[-2], file_names_int)

csv_list=glob('/home/zzg/data/micro_expression/CASME^2_longVideoFaceCropped/features_engineered/predictions/*/*/*.csv')
df_predict = pd.DataFrame(data=sorted(csv_list, key=sort_fun), columns=['file_path'])
df_predict.to_csv('/home/zzg/data/micro_expression/CASME^2_longVideoFaceCropped/features_engineered/predict_facecroped_vad.csv', index=False)