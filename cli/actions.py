import json
import os
import pickle
import time
from pathlib import Path
from typing import List, cast
import numpy as np
import tabulate
import yaml

from cli.interface import ModelLocation
from config import app_settings
from config.interface import TrainingProjectFileNames, HistorySummary

def _calc_dir_size(path: str):
    root_dir = Path(path)
    dir_size = sum(f.stat().st_size for f in root_dir.glob("**/*.*") if f.is_file())
    return dir_size


def _get_history(name: str, session_id: str) -> HistorySummary:
    """writing new logic instead of using TrainingProject because TrainingProject requires loading tensorflow lib"""
    path_history = os.path.join(
        app_settings.dir_training, name, session_id, TrainingProjectFileNames.HISTORY_SUMMARY.value)
    with open(path_history, "rb") as f:
        return pickle.load(f, fix_imports=False)


def list_models(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    result: List[ModelLocation] = []
    names = os.listdir(app_settings.dir_training)
    for name in names:
        for session_id in os.listdir(os.path.join(app_settings.dir_training, name)):
            model_path = os.path.join(app_settings.dir_training, name, session_id)
            history = _get_history(name=name, session_id=session_id)
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
    from config.training_project import TrainingProject
    project = TrainingProject(name=name, session_id=session_id)
    title = "Model Architecture: {0}".format(project.path_training_dir)
    print(title)
    print("_"*len(title))
    print()
    print(project.model.summary())


def load_description_file(path: str):
    _, ext = os.path.splitext(path)
    with open(path, "r") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".yaml" or ext == ".yml":
            return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Description loader for extension '{0}' not found".format(ext))


def train_model(args):
    path = args.file if os.path.isabs(args.file) else os.path.abspath(args.file)
    description = load_description_file(path)
    from workflows.main_training_workflow import train
    train(description)

