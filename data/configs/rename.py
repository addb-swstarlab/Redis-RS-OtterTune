import os

path = "/home/capstone2201/new-redis-sample-generation/redis-sample-generation/configfile/"
for file_name in os.listdir(path):
    if ".conf" in file_name and "redis" not in file_name:
        idx = int(file_name[6:-5])
        if not(1 <= idx <= 200 or 10001 <= idx <= 10200):
            os.remove(path + file_name)
        # with open(path + file_name, 'r') as f:
        #     lines = f.readlines() 
        # with open(path + file_name, 'w') as f:
        #     for line in lines:
        #         if line.strip('\n') == "dir /home/juyeon/redis-logs":
        #             f.write("dir /home/capstone2201/redis-logs\n")
        #         else:
        #             f.write(line)