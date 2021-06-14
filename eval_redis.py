import xmlrpc.client
import time
import http.client
import argparse
import os

class TimeoutTransport(xmlrpc.client.Transport):
    timeout = 30.0

    def set_timeout(self, timeout):
        self.timeout = timeout

    def make_connection(self, host):
        h = http.client.HTTPConnection(host, timeout=self.timeout)
        return h

parser = argparse.ArgumentParser()
parser.add_argument('--persistence', type=str, choices=["RDB","AOF"],default='RDB', help='Choose Persistant Methods')
parser.add_argument('--modelnum', type=int, default='0', help='Choose Model number')

opt = parser.parse_args()

server_ip = '10.178.0.6'
transport = TimeoutTransport()
transport.set_timeout(60)

CONFIG_PATH = 'data/redis_data/config_results/{}'.format(opt.persistence)
FILE_PATH = '{}_rec_config{}.conf'.format(opt.persistence,str(opt.modelnum))

s = xmlrpc.client.ServerProxy('http://%s:20000' % server_ip, transport=transport)
f = open(os.path.join(CONFIG_PATH,FILE_PATH))
config = f.readlines()
try:
	s.start_redis('redis',config)
except xmlrpc.client.Fault as err:
	print('except: client Fault')
	print('Fault code: %d' % err.faultCode)
	print('Fault string: %s' % err.faultString)
	time.sleep(5)
