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
    

def start_redis(instance_name, configs):
# def start_redis(instance_name):
    # params = configs.split(',')
    params = configs
    write_cnf_file(params)
    
    # To do: start redis server
    os.system('/home/jinhuijun/CDBTune/redis-5.0.2/src/redis-cli shutdown')
    os.system('rm -rf /home/jinhuijun/CDBTune/redis-5.0.2/appendonly.aof')
    os.system('rm -rf /home/jinhuijun/CDBTune/redis-5.0.2/dump.rdb')
    print('finish os')
    sudo_exec('sudo echo 3 > /proc/sys/vm/drop_caches', '1031')
    os.system('/home/jinhuijun/CDBTune/redis-5.0.2/src/redis-server /home/jinhuijun/CDBTune/redis-5.0.2/redis.conf')
    # sudo_exec('sudo service redis restart', '1031')
    return 1


# def write_cnf_file(configs):
#     """
#     Args:
#         configs: str, Formatted MySQL Parameters, e.g. "--binlog_size=xxx"
#     """
#     conf_file = '../redis-5.0.2/redis.conf'
    
#     cnf_file = transformcfg_redis_to_ini(conf_file, 'redis') # ../redis-5.0.2/redis.cnf
    
#     config_parser = CP.ConfigParser()
#     print("#### chmod cnf file to 777")
#     sudo_exec('sudo chmod 777 %s' % cnf_file, '1031')
#     time.sleep(2)
#     config_parser.read(cnf_file)
#     for param in configs:
#         pair_ = param.split(':')
#         config_parser.set('redis', pair_[0], pair_[1])
#     config_parser.write(open(cnf_file, 'w'))
#     print("finished write config_parser")
#     conf_file = transformcfg_ini_to_redis(cnf_file)
#     print("finished write transformcfg_ini_to_redis")
    
#     sudo_exec('sudo chmod 744 %s' % cnf_file, '1031')
#     time.sleep(2)


def write_cnf_file(config):
    conf_file_path = '../redis-5.0.2/redis.conf'
    
    print("#### chmod cnf file to 777")
    sudo_exec('sudo chmod 777 %s' % conf_file_path, '1031')
    with open(conf_file_path, 'w') as f:
        f.writelines(config)
    print("finished write recommended configuration")

    sudo_exec('sudo chmod 744 %s' % conf_file_path, '1031')
    time.sleep(2)


def serve():
    server = SimpleXMLRPCServer(('0.0.0.0', 20000))
    server.register_function(start_redis)
    try:
        print('Use Control-C to exit')
        server.serve_forever()
    except KeyboardInterrupt:
        print('Exiting')


def transformcfg_redis_to_ini(conf_file, db_name):
    f = open(conf_file,'r')
    cnf = []
    cnf.append("[%s]\n"%db_name) # the case in redis DB
    ep = ['#', '\n']

    while True:
        line = f.readline()
        if not line: break
        if line[0] not in ep:
            if line.split(' ')[0] == 'save': continue
            if line.split(' ')[0] == 'client-output-buffer-limit': continue
            cnf_line = line.split(' ')[0] + ' = ' + line.split(' ')[1]
            print('cnf_line: ', cnf_line)
            cnf.append(cnf_line)

    save_cnf_file = conf_file[:-3] + 'nf' # redis.conf ==> redis.cnf
    print("make new cnf file for redis")
    with open(save_cnf_file, 'w') as cf:
        cf.writelines(cnf)
    return save_cnf_file

def transformcfg_ini_to_redis(cnf_file):
    f = open(cnf_file, 'r')
    cnf = f.readlines()
    cnf = cnf[:-1]
    conf = []

    print('start iteration for')
    print(len(cnf))
    for i in range(len(cnf)):
        if i == 0: continue
        print(i)
        conf_line = cnf[i].split(' = ')[0] + ' ' + cnf[i].split(' = ')[1]
        print(conf_line)
        conf.append(conf_line)

    save_conf_file = cnf_file[:-2] + 'onf' # redis.cnf ==> redis.conf
    print('save_conf_file: ', save_conf_file)
    with open(save_conf_file, 'w') as cf:
        cf.writelines(conf)
    return save_conf_file

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--docker', action='store_true')
    # opt = parser.parse_args()
    # if opt.docker:
    #     docker = False

    serve()


