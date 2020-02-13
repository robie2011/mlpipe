import logging
import os
import time
from pathlib import Path
from typing import List, cast
from mlpipe.config import app_settings
# some imports are done withing functions for performance improvements
from mlpipe.dsl_interpreter.interpreter import create_workflow_from_file
from mlpipe.workflows.utils import load_description_file

module_logger = logging.getLogger(__name__)


def _print_heading(text: str):
    line = f"MLPIPE CLI: {text}"
    module_logger.info(line)


def _calc_dir_size(path: str):
    root_dir = Path(path)
    dir_size = sum(f.stat().st_size for f in root_dir.glob("**/*.*") if f.is_file())
    return dir_size


def _get_history(name: str, session_id: str):
    from mlpipe.config.interface import TrainingProjectFileNames
    import pickle
    # todo: check
    # code duplication: (TrainingProject)
    # because TrainingProject requires loading tensorflow lib
    path_history = os.path.join(
        app_settings.dir_training, name, session_id, TrainingProjectFileNames.HISTORY_SUMMARY.value)
    if not os.path.isfile(path_history):
        return None

    with open(path_history, "rb") as f:
        return pickle.load(f, fix_imports=False)


def list_models(args):
    import numpy as np
    import tabulate
    from mlpipe.cli.interface import ModelLocation
    _print_heading("LIST")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    result: List[ModelLocation] = []
    names = os.listdir(app_settings.dir_training)
    for name in names:
        for session_id in os.listdir(os.path.join(app_settings.dir_training, name)):
            model_path = os.path.join(app_settings.dir_training, name, session_id)
            history = _get_history(name=name, session_id=session_id)
            if history is None:
                print("history not found in training folder: {0}".format(model_path))
                continue

            ix_best = np.argmin(history.history[app_settings.training_monitor])

            result.append(ModelLocation(
                name=name,
                session_id=session_id,
                path=model_path,
                sizeBytes=_calc_dir_size(model_path),
                monitored_value=history.history[app_settings.training_monitor][ix_best],
                accuracy=history.history['accuracy'][ix_best],
                epochs=len(history.epoch),
                batch_size=history.params['batch_size'],
                samples=history.params['samples'],
                metrics=", ".join(history.params['metrics']),
                datetime=time.ctime(os.path.getmtime(model_path))
            ))

    result = sorted(result, key=lambda r: r.accuracy, reverse=True)
    result = sorted(result, key=lambda r: r.name)

    print(tabulate.tabulate(map(
        lambda x: [
            "{0}/{1}".format(x.name, x.session_id),
            x.sizeBytes / 1024,
            x.monitored_value,
            x.accuracy,
            x.epochs,
            x.batch_size,
            x.samples,
            x.datetime
        ], result),
        headers=["name/session", "size (KB)", app_settings.training_monitor, 'accuracy', "epochs", "batch_size",
                 "samples", "datetime"]))


def describe_model(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    name, session_id = cast(str, args.model_session).split("/")
    from mlpipe.config.training_project import TrainingProject
    project = TrainingProject(name=name, session_id=session_id)
    _print_heading("DESCRIBE MODEL")
    title = "Model Architecture: {0}".format(project.path_training_dir)
    print(title)
    print("_" * len(title))
    print()
    print(project.model.summary())


def train_model(args):
    from mlpipe.dsl_interpreter.interpreter import create_workflow_from_file

    for f in args.files:
        path = f if os.path.isabs(f) else os.path.abspath(f)
        description = load_description_file(path)
        _print_heading(f"TRAINING MODEL: {description['name']}")
        module_logger.info(f"file: {path}")
        manager = create_workflow_from_file(path, overrides={"@mode": "train"})
        path_training_dir, model = manager.run()
        module_logger.info(f"trained model: {path_training_dir}")


def integrate_model(args):
    import threading
    threads = []

    for f in args.files:
        path = f if os.path.isabs(f) else os.path.abspath(f)
        description = load_description_file(path)
        _print_heading(f"INTEGRATE MODEL: {description['name']}/{description['session']}")
        module_logger.info(f"file: {path}")
        manager = create_workflow_from_file(path, overrides={"@mode": "integrate"})
        t = threading.Thread(target=manager.run)
        threads.append(t)
        t.start()

    module_logger.info(f"Total threads: {len(threading.enumerate())}")


def test_model(args):
    for f in args.files:
        try:
            path = f if os.path.isabs(f) else os.path.abspath(f)
            description = load_description_file(path)

            _print_heading("TESTING MODEL: {0}/{1}".format(description['name'], description['session']))
            module_logger.info(f"file: {path}")
            module_logger.info("test data source: ")
            for k, v in description['source'].items():
                if not isinstance(v, str) and v.__iter__:
                    module_logger.info("   {0}: ".format(k))
                    for x in v:
                        module_logger.info("   - {0}".format(x))
                else:
                    module_logger.info("   {0}: {1}".format(k, v))
            module_logger.info("")

            description['@mode'] = 'evaluate'
            manager = create_workflow_from_file(path, overrides={"@mode": "evaluate"})
            result = manager.run()
            for k, v in result.items():
                print(f"     {k}: {v}")

        except Exception as e:
            module_logger.error(e)


# def print_evaluation_result(result):
#     module_logger.info("")
#     module_logger.info("result:")
#     terminal_tab = "   "
#     for attr, value in result.__dict__.items():
#         # formatting number output: https://pyformat.info/
#         if attr.startswith("n_"):
#             module_logger.info(terminal_tab + "{0}: {1} {2:05.3f}%".format(attr, value, value / result.size * 100))
#         elif attr.startswith("p_"):
#             module_logger.info(terminal_tab + "{0}: {1:05.3f}%".format(attr, value * 100))
#         elif isinstance(value, numbers.Number):
#             module_logger.info(terminal_tab + "{0}: {1:,}".format(attr, value))
#         elif isinstance(value, dict):
#             module_logger.info(terminal_tab + "{0}:".format(attr))
#             for k, v in value.items():
#                 if isinstance(v, tuple):
#                     module_logger.info(terminal_tab * 2 + "{0}:".format(k))
#                     for ti in v:
#                         module_logger.info(terminal_tab * 3 + "{0}".format(ti))
#                 else:
#                     module_logger.info(terminal_tab * 2 + "{0}: {1}".format(k, v))
#         else:
#             #module_logger.info("   {0}: ".format(str(attr)), value)
#             module_logger.info(terminal_tab + attr + ":")
#             module_logger.info(value)


def analyze_data(args):
    from mlpipe.config.analytics_data_manager import AnalyticsDataManager

    if args.create:
        import grequests
        name_updates = []
        for f in args.files:
            _print_heading("ANALYZE")
            path = f if os.path.isabs(f) else os.path.abspath(f)
            base_path, ext = os.path.splitext(path)
            name = os.path.basename(base_path)

            description = load_description_file(path)
            keys_allowed = ['source', 'analyze', 'pipeline']
            filtered_desc = dict(filter(lambda x: x[0] in keys_allowed, description.items()))
            AnalyticsDataManager.save(name=name, description=filtered_desc, overwrite=args.force)
            name_updates.append(name)

        # send signal to webserver
        rs = [grequests.post(f"http://localhost:{app_settings.api_port}/api/signal",
                             data={"type": "analytics_description", "name": n})
              for n in name_updates]
        grequests.map(rs)

    if args.list:
        AnalyticsDataManager.list_files()
        return

    if args.delete:
        for name in args.files:
            AnalyticsDataManager.delete(name)
        return
