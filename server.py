from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from view_test import excute_detect
odata = {'result': 'this is a test'}
host = ('localhost', 8888)


class Resquest (BaseHTTPRequestHandler):

    def do_GET (self):
        print('这是请求数据')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(odata).encode())

    def do_POST (self):
        print('这是请求数据', self.path[7:])
        self.send_response(200)
        # content_length = int(self.headers['Content-Length'])
        # print(self.rfile.read(content_length))
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        trueTap, tap = excute_detect(self.path[7:])
        self.end_headers()
        data = {}
        data['data'] = tap
        data['origin'] = trueTap
        self.wfile.write(json.dumps(data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

