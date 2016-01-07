

from flask import Flask, Response, render_template, request, send_from_directory
from flask.ext.cors import CORS

import json
import uuid



def result_ok(result):
    return {"status":True, "result":result}

def result_error(msg):
    return {"status":False, "msg":msg}

class snakefoot_Server():

    def __init__(self, app_instance):
        global app
        app = app_instance
        self.app = app

    def set_rpc_instance(self, instance):
        self.rpc_instance = instance

    def index(self):
        index_callback = getattr(self.rpc_instance, "index", False)
        if index_callback:
            index_callback()
        else:
            return

    @property
    def index_page(self):
        return self.rpc_instance.index_page
    
    @index_page.setter
    def index_page(self, val):
        return

    def invoke(self, req):
        callback_name = req["callback_name"]
        callback_params = req["callback_params"]
        callback = getattr(self.rpc_instance, callback_name, None)
        
        if callback is None:
            return result_error("error:no such callback")

        callback_result = None
        callback_result = callback(callback_params)
        # try:
        #     callback_result = callback(callback_params)
        # except Exception as e:
        #     raise e
        #     return result_error("error:"+str(e))

        if callback_result is None:
            callback_result = {}
            
        return result_ok(callback_result)

    def bind(self, host="localhost", port=10088, debug=False):
        self.host = host
        self.port = port
        self.debug = debug

    def run(self):
        app.run(host=self.host, port=self.port, debug=self.debug)




def response_ok(result):
    return json.dumps(result, ensure_ascii=False)

def response_error(error_msg):
    result = result_error(error_msg)
    return json.dumps(result, ensure_ascii=False)

def gen_uuid():
    return str(uuid.uuid1())


def Server(rpc_instance):
    app = Flask(__name__)
    CORS(app)

    snakefoot_server = snakefoot_Server(app)
    snakefoot_server.set_rpc_instance(rpc_instance)

    @app.route("/", methods=["GET"])
    def index():
        snakefoot_server.index()
        return render_template(snakefoot_server.index_page)

    @app.route("/__snakefoot_rpc_invoke", methods=["POST"])
    def snakefoot_rpc_invoke():
        try:
            data = request.data
            req = json.loads(data)
        except:
            return response_error("data type error")

        result = snakefoot_server.invoke(req)
        return response_ok(result)

    @app.route('/js/<path:path>')
    def send_js(path):
        return send_from_directory('./static/js', path)

    @app.route('/css/<path:path>')
    def send_css(path):
        return send_from_directory('./static/css', path)

    @app.route('/image/<path:path>')
    def send_image(path):
        return send_from_directory('./static/image', path)
    
    return snakefoot_server





