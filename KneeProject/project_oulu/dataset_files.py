import os
import numpy as np
import pandas as pd
import time

subfolder = 'test'
save_dir = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_oulu/'

save_dir = save_dir + subfolder

print(os.listdir(save_dir))
dataset = []
for each in os.listdir(save_dir):
    label = each
    save_dir2 = os.path.join(save_dir,each)

    for png_file in os.listdir(save_dir2):
        dataset.append((os.path.join(save_dir2,png_file),label))


df = pd.DataFrame(dataset,columns=['directory','label'])
df.to_csv(subfolder + '.csv',index=False)
print(df)
print(df.shape)
