import xmlrpc.client
import time
import sys
import http.client


class TimeoutTransport(xmlrpc.client.Transport):
    timeout = 30.0

    def set_timeout(self, timeout):
        self.timeout = timeout

    def make_connection(self, host):
        h = http.client.HTTPConnection(host, timeout=self.timeout)
        return h


server_ip = '10.178.0.6'
transport = TimeoutTransport()
transport.set_timeout(60)

s = xmlrpc.client.ServerProxy('http://%s:20000' % server_ip, transport=transport)
configs = ["threads:2"]
try:
	s.start_rocksdb('rocksdb', configs)
except xmlrpc.client.Fault as err:
    print('except: client Fault')
    print("Fault code: %d" % err.faultCode)
    print("Fault string: %s" % err.faultString)
    time.sleep(5)
