import os
import shutil

print(os.listdir('./data/redis_data/config2'))

CONFIG_PATH = './data/redis_data/config2'
new_path = './data/redis_data/configs2'
count = 1
for i in range(1,19):
    for j in range(1001,2001):
        shutil.copyfile(os.path.join(CONFIG_PATH,f'workload{i}',f'config{j}.conf'),os.path.join(new_path,f'config{count}.conf'))
        count+=1

        
