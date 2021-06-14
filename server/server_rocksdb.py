# -*- coding: utf-8 -*-
"""
Configure Server
"""

import os
import time
import pexpect
import platform
import argparse
import configparser as CP
from xmlrpc.server import SimpleXMLRPCServer

docker = False


# def get_state():
#     check_start()
#     m = os.popen('service redis status')
#     s = m.readlines()[2]
#     s = s.split(':')[1].replace(' ', '').split('(')[0]
#     if s == 'failed':
#         return -1
#     return 1


# def check_start():
#     a = sudo_exec('sudo tail -1 /var/log/mysql/ubunturmw.err', '123456')
#     a = a.strip('\n\r')
#     if a.find('pid ended') != -1:
#         sudo_exec('sudo service mysql start', '123456')


def sudo_exec(cmdline, passwd):
    osname = platform.system()
    if osname == 'Linux':
        prompt = r'\[sudo\] password for %s: ' % os.environ['USER']
    elif osname == 'Darwin':
        prompt = 'Password:'
    else:
        assert False, osname
    child = pexpect.spawn(cmdline)
    idx = child.expect([prompt, pexpect.EOF], 3)
    if idx == 0:
        child.sendline(passwd)
        child.expect(pexpect.EOF)
    return child.before

def cus_exec(cmdline):
    child = pexpect.spawn(cmdline)
    return child.before
    

def start_rocksdb(instance_name, configs):
    # params = configs.split(',')

    params = configs
    cmdline = write_cnf_file(params)
    
    # To do: start db_bench for rocksdb
    os.system(cmdline)
    return 1


def write_cnf_file(configs):
    """
    Args:
        configs: str, Formatted MySQL Parameters, e.g. "--binlog_size=xxx"
    """
    cnf_file = '../rocksdb/rocksdb.cnf'
    
    config_parser = CP.ConfigParser()
    print("#### chmod cnf file to 777")
    sudo_exec('sudo chmod 777 %s' % cnf_file, '1031')
    time.sleep(2)
    config_parser.read(cnf_file)
    for param in configs:
        pair_ = param.split(':')
        config_parser.set('rocksdb', pair_[0], pair_[1])
    config_parser.write(open(cnf_file, 'w'))
    
    sudo_exec('sudo chmod 744 %s' % cnf_file, '1031')
    time.sleep(2)
    
    return write_db_bench_cmdline(cnf_file)

def write_db_bench_cmdline(cnf_file):
    f = open(cnf_file, 'r')
    rock_cnf = f.readlines()

    rocks_cmd_line = '../rocksdb/db_bench '
    for i in range(len(rock_cnf)):
        if i==0: continue
        rocks_cmd_line += '--' + ''.join(rock_cnf[i].split()) + ' '

    return rocks_cmd_line[:-1]


def serve():
    server = SimpleXMLRPCServer(('0.0.0.0', 20000))
    server.register_function(start_rocksdb)
    try:
        print('Use Control-C to exit')
        server.serve_forever()
    except KeyboardInterrupt:
        print('Exiting')




if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--docker', action='store_true')
    # opt = parser.parse_args()
    # if opt.docker:
    #     docker = False

    serve()


