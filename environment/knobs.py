# -*- coding: utf-8 -*-
"""
desciption: Knob information

"""

# 700GB
memory_size = 360*1024*1024
#
disk_size = 8*1024*1024*1024
instance_name = ''


KNOBS = [#'skip_name_resolve',               # OFF
         'table_open_cache',                # 2000
         #'max_connections',                 # 151
         'innodb_buffer_pool_size',         # 134217728
         'innodb_buffer_pool_instances',    # 8
         #'innodb_log_files_in_group',       # 2
         #'innodb_log_file_size',            # 50331648
         'innodb_purge_threads',            # 1
         'innodb_read_io_threads',          # 4
         'innodb_write_io_threads',         # 4
         #'binlog_cache_size',               # 32768
         #'max_binlog_cache_size',           # 18446744073709547520
         #'max_binlog_size',                 # 1073741824
         ]

KNOB_DETAILS = None
EXTENDED_KNOBS = None
num_knobs = len(KNOBS)



def get_init_knobs():

    knobs = {}
    
    for name, value in KNOB_DETAILS.items():
        knob_value = value[1]
        knobs[name] = knob_value[-1]

    return knobs


def gen_continuous(action):
    knobs = {}

    for idx in xrange(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]

        knob_type = value[0]
        knob_value = value[1]
        min_value = knob_value[0]

        if knob_type == 'integer':
            max_val = knob_value[1]
            eval_value = int(max_val * action[idx])
            eval_value = max(eval_value, min_value)
        else:
            enum_size = len(knob_value)
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = knob_value[enum_index]


        knobs[name] = eval_value


    return knobs


def save_knobs(knob, metrics, knob_file):
    """ Save Knobs and their metrics to files
    Args:
        knob: dict, knob content
        metrics: list, tps and latency
        knob_file: str, file path
    """
    # format: tps, latency, knobstr: [#knobname=value#]
    knob_strs = []
    for kv in knob.items():
        knob_strs.append('{}:{}'.format(kv[0], kv[1]))
    result_str = '{},{},{},'.format(metrics[0], metrics[1], metrics[2])
    knob_str = "#".join(knob_strs)
    result_str += knob_str

    with open(knob_file, 'a+') as f:
        f.write(result_str+'\n')

