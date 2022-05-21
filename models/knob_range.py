import json

name = ['hash-max-ziplist-entries', 'hash-max-ziplist-value', 'activerehashing',
 'hz', 'dynamic-hz', 'save1_sec', 'save1_changes', 'save2_sec', 'save2_changes',
 'save3_sec','save3_changes', 'rdbcompression', 'rdbchecksum',
 'rdb-save-incremental-fsync', 'activedefrag',
 'active-defrag-threshold-lower', 'active-defrag-threshold-upper',
 'active-defrag-cycle-min', 'active-defrag-cycle-max']

minval = [100, 8, 0, 1, 0, 700, 1, 100, 10, 10, 7500, 0, 0, 0, 0, 5, 50, 5, 50]

maxval = [1500, 256, 1, 200, 1, 1400, 9, 699, 100, 99, 17500, 1, 1, 1, 1, 49, 100, 49, 100]

default = [512, 64, 1, 10, 1, 900, 1, 300, 10, 60, 10000, 1, 1, 1, 1, 10, 100, 5, 75]

RDB_Knobs = []

for i in range(len(name)):
    RDB_knob = {}
    RDB_knob['name'] = name[i]
    RDB_knob['minval'] = minval[i]
    RDB_knob['maxval'] = maxval[i]
    RDB_knob['default'] = default[i]
    RDB_Knobs.append(RDB_knob)

<<<<<<< HEAD
with open('/home/capstone2201/data/RDB_knobs.json', 'w') as j: #수정
=======
with open('../data/RDB_knobs.json', 'w') as j: #수정
>>>>>>> 193a39b3947beb719ab7abb674ecc477fa7e9892
    json.dump(RDB_Knobs, j)