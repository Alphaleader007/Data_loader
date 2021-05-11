import os
import pandas as pd

def Get(x):
    data_csv_file = "../input/siim-isic-melanoma-classification/train.csv"
    df_spec = pd.read_csv(data_csv_file, usecols=['image_name', 'target'])
    im=x
    l = im.split(os.path.sep)[-1]
    label = l[:len(l) - 4]
    label
    def remove(string):
        return string.replace(" ", "")
    df2 = df_spec[(df_spec.image_name == label) ]
    x=df2['target']
    y=str(x)
    y= y[5:len(y) - 27]    
    y=remove(y)
    global Pred
    Pred="tensor([{}], device='cuda:0')".format(y)