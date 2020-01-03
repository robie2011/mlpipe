from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from mlpipe.config import app_settings
from mlpipe.config.analytics_data_manager import AnalyticsDataManager
import datetime
import logging
from mlpipe.workflows.analyzers.create_analyzers import create_analyzer_workflow
from mlpipe.workflows.load_data.create_loader import create_loader_workflow
from mlpipe.workflows.pipeline.create_pipeline import create_pipeline_workflow


module_logger = logging.getLogger(__name__)
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')
start_time = datetime.datetime.now()


@app.route('/')
def hello_world():
    return 'MLPIPE Webserver online since ' +start_time


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
        data = create_pipeline_workflow(descriptions=desc['pipeline']).execute(data)

    result = create_analyzer_workflow(desc['analyze']).run(data)
    return jsonify(result)


@app.route('/api/signal', methods=['POST'])
def signal():
    print("signal received", request.form)
    # works

    #socketio.emit('my response', request.form, broadcast=True, namespace='/ws')
    socketio.send(request.form, broadcast=True, namespace='/ws')
    return "OK"


if __name__ == '__main__':
    #app.run(port=app_settings.api_port)
    socketio.run(app, port=app_settings.api_port, debug=True)
