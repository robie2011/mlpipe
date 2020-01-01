import json
import numbers
import os
import time
from pathlib import Path
from typing import List, cast
import logging
from mlpipe.config import app_settings
# some imports are done withing functions for performance improvements


module_logger = logging.getLogger(__name__)


def _calc_dir_size(path: str):
    root_dir = Path(path)
    dir_size = sum(f.stat().st_size for f in root_dir.glob("**/*.*") if f.is_file())
    return dir_size


def _get_history(name: str, session_id: str):
    from mlpipe.config.interface import TrainingProjectFileNames, HistorySummary
    import pickle
    """writing new logic instead of using TrainingProject because TrainingProject requires loading tensorflow lib"""
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
            x.sizeBytes/1024,
            x.monitored_value,
            x.accuracy,
            x.epochs,
            x.batch_size,
            x.samples,
            x.datetime
        ], result),
        headers=["name/session", "size (KB)", app_settings.training_monitor, 'accuracy', "epochs", "batch_size", "samples", "datetime"]))


def describe_model(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    name, session_id = cast(str, args.model_session).split("/")
    from mlpipe.config.training_project import TrainingProject
    project = TrainingProject(name=name, session_id=session_id)
    title = "Model Architecture: {0}".format(project.path_training_dir)
    print(title)
    print("_"*len(title))
    print()
    print(project.model.summary())


def _load_description_file(path: str):
    import yaml
    _, ext = os.path.splitext(path)
    with open(path, "r") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".yaml" or ext == ".yml":
            return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Description loader for extension '{0}' not found".format(ext))


def train_model(args):
    from mlpipe.workflows.main_training_workflow import train

    for f in args.files:
        path = f if os.path.isabs(f) else os.path.abspath(f)
        description = _load_description_file(path)
        module_logger.info("")
        module_logger.info(f"TRAINING MODEL: {description['name']}")
        module_logger.info(f"file: {path}")
        train(description)


def test_model(args):
    from mlpipe.workflows.main_evaluate_workflow import evaluate
    for f in args.files:
        try:
            path = f if os.path.isabs(f) else os.path.abspath(f)
            description = _load_description_file(path)

            module_logger.info("")
            module_logger.info("TESTING MODEL: {0}/{1}".format(description['name'], description['session']))
            module_logger.info(f"file: {path}")
            module_logger.info("test data source: ")
            for k, v in description['testSource'].items():
                if not isinstance(v, str) and v.__iter__:
                    module_logger.info("   {0}: ".format(k))
                    for x in v:
                        module_logger.info("   - {0}".format(x))
                else:
                    module_logger.info("   {0}: {1}".format(k, v))
            module_logger.info("")

            result = evaluate(description)
            print_evaluation_result(result)
        except Exception as e:
            module_logger.error(e)


def print_evaluation_result(result):
    module_logger.info("")
    module_logger.info("result:")
    terminal_tab = "   "
    for attr, value in result.__dict__.items():
        # formatting number output: https://pyformat.info/
        if attr.startswith("n_"):
            module_logger.info(terminal_tab + "{0}: {1} {2:05.3f}%".format(attr, value, value / result.size * 100))
        elif attr.startswith("p_"):
            module_logger.info(terminal_tab + "{0}: {1:05.3f}%".format(attr, value * 100))
        elif isinstance(value, numbers.Number):
            module_logger.info(terminal_tab + "{0}: {1:,}".format(attr, value))
        elif isinstance(value, dict):
            module_logger.info(terminal_tab + "{0}:".format(attr))
            for k, v in value.items():
                if isinstance(v, tuple):
                    module_logger.info(terminal_tab * 2 + "{0}:".format(k))
                    for ti in v:
                        module_logger.info(terminal_tab * 3 + "{0}".format(ti))
                else:
                    module_logger.info(terminal_tab * 2 + "{0}: {1}".format(k, v))
        else:
            module_logger.info("   {0}: ".format(attr), value)


def analyze_data(args):
    from mlpipe.config.analytics_data_manager import AnalyticsDataManager
    if args.create:
        for f in args.files:
            path = f if os.path.isabs(f) else os.path.abspath(f)
            base_path, ext = os.path.splitext(path)
            name = os.path.basename(base_path)

            description = _load_description_file(path)
            keys_allowed = ['source', 'analyze']
            filtered_desc = dict(filter(lambda x: x[0] in keys_allowed, description.items()))
            #create_analyzer_workflow(filtered_desc).run()
            AnalyticsDataManager.save(name=name, description=filtered_desc, overwrite=args.force)
            return
    if args.list:
        AnalyticsDataManager.list_files()
        return

    if args.delete:
        for name in args.files:
            AnalyticsDataManager.delete(name)
        return
