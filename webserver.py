import datetime
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from mlpipe.config import app_settings
from mlpipe.config.analytics_data_manager import AnalyticsDataManager
from mlpipe.workflows.analyzers.main_analyze_workflow import create_analyzer_workflow
from mlpipe.workflows.load_data.create_loader import create_loader_workflow
from mlpipe.workflows.pipeline.pipeline_builder import build_pipeline_executor

module_logger = logging.getLogger(__name__)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
start_time = datetime.datetime.now()
logging.getLogger('flask_cors').level = logging.DEBUG
cors = CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/socket.io/*": {"origins": "*"}
})


@app.route('/')
def hello_world():
    return 'MLPIPE Webserver online since ' + str(start_time)


@app.route('/api/analytics_descriptions')
def list_analytics_descriptions():
    files = AnalyticsDataManager.list_files(suppress_output=True)
    return jsonify(files)


@app.route('/api/analytics_description/<string:name>')
def view_analytics_description(name: str):
    return AnalyticsDataManager.get(name)


@app.route('/api/analytics/<string:name>')
def view_analytics(name: str):
    module_logger.debug("load description for " + name)
    desc = AnalyticsDataManager.get(name)

    module_logger.debug("load data")
    data = create_loader_workflow(description=desc['source']).load()

    if 'pipeline' in desc:
        module_logger.debug("pipe data")
        data = build_pipeline_executor(descriptions=desc['pipeline']).execute(data)

    result = create_analyzer_workflow(desc['analyze']).run(data)
    #return json.dumps(result, allow_nan=False, default=lambda o: o.__dict__)
    # todo: this allows nan!
    return jsonify(result)


@app.route('/api/signal', methods=['POST'])
def signal():
    print("signal received", request.form)
    # expected input example: {"type": "analytics_description", "name": "someAnalyticsFileNameWithoutExt"}
    socketio.send(request.form, broadcast=True, namespace='/ws')
    return "OK"


@socketio.on('connect', namespace='/ws')
def test_connect():
    socketio.send({'data': 'Connected'})


@socketio.on('disconnect', namespace='/ws')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    #app.run(port=app_settings.api_port)
    socketio.run(app, port=app_settings.api_port, debug=True)
