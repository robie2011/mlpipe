import logging
import os
from os.path import basename
from pathlib import Path
from typing import List, cast

# some imports are done withing functions for performance improvements
from mlpipe.config.app_settings import AppConfig
from mlpipe.dsl_interpreter.interpreter import create_workflow_from_file
from mlpipe.utils.file_handlers import write_text_file
from mlpipe.workflows.analyze.create_report import generate_html_report
from mlpipe.workflows.analyze.interface import AnalyticsResult
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
        AppConfig['training.dir_data'], name, session_id, TrainingProjectFileNames.HISTORY_SUMMARY.value)
    if not os.path.isfile(path_history):
        return None

    with open(path_history, "rb") as f:
        return pickle.load(f, fix_imports=False)


# noinspection PyUnusedLocal
def list_models(args):
    import tabulate
    from mlpipe.cli.interface import ModelLocation
    from datetime import datetime
    _print_heading("LIST")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    result: List[ModelLocation] = []
    names = os.listdir(AppConfig['training.dir_data'])
    for name in names:
        for session_id in os.listdir(os.path.join(AppConfig['training.dir_data'], name)):
            model_path = os.path.join(AppConfig['training.dir_data'], name, session_id)
            history = _get_history(name=name, session_id=session_id)
            if history is None:
                print("history not found in training folder: {0}".format(model_path))
                continue

            result.append(ModelLocation(
                name=name,
                session_id=session_id,
                path=model_path,
                sizeBytes=_calc_dir_size(model_path),
                epochs=len(history.epoch),
                batch_size=history.params['batch_size'],
                samples=history.params['samples'],
                metrics=", ".join(history.params['metrics']),
                datetime=datetime.fromtimestamp(os.path.getmtime(model_path))
            ))

    result = sorted(result, key=lambda r: r.datetime, reverse=True)

    print(tabulate.tabulate(map(
        lambda x: [
            "{0}/{1}".format(x.name, x.session_id),
            x.sizeBytes / 1024,
            x.epochs,
            x.batch_size,
            x.samples,
            x.datetime
        ], result),
        headers=["name/session", "size (KB)", "epochs", "batch_size",
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
    f = args.file
    path = f if os.path.isabs(f) else os.path.abspath(f)
    description = load_description_file(path)
    _print_heading(f"INTEGRATE MODEL: {description['name']}/{description['session']}")
    module_logger.info(f"file: {path}")
    manager = create_workflow_from_file(path, overrides={"@mode": "integrate"})
    manager.run()


def evaluate_model(args):
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

            # NOTE:
            #   score is evaluation of given loss function
            #   accuracy is accuracy of given function

            for k, v in result.__dict__.items():
                module_logger.info(f"{k}: {v}")

        except Exception as e:
            module_logger.error(e)


def analyze_data(args):
    import simplejson
    import os

    for f in args.files:
        _print_heading("ANALYZE")
        desc_file = f if os.path.isabs(f) else os.path.abspath(f)
        result: AnalyticsResult = create_workflow_from_file(desc_file, overrides={"@mode": "analyze"}).run()
        data = simplejson.dumps(result, ignore_nan=True, default=lambda o: o.__dict__)
        file_basename = '.'.join(basename(desc_file).split('.')[:-1])
        output_folder = Path(os.path.dirname(desc_file))
        output_file = output_folder / f"report_{file_basename}.html"
        output_file_json = output_folder / f"report_{file_basename}.json"
        print(f"writing report: {output_file}")
        generate_html_report(json_str=data, output_path=output_file)

        if args.json:
            write_text_file(output_file_json, data)


def export_data(args):
    import os
    import pandas as pd

    overrides = {
        "@mode": "export",
        "@pipelinePrimary": True if (args.pipelinePrimary or args.full) else False,
        "@pipelineSecondary": True if args.full else False
    }

    suffix = "source"

    if overrides['@pipelinePrimary']:
        suffix = "primary"

    if overrides['@pipelineSecondary']:
        suffix = "full"

    for f in args.files:
        _print_heading("EXPORT")
        desc_file = f if os.path.isabs(f) else os.path.abspath(f)
        desc_file = Path(desc_file)
        result: pd.DataFrame = create_workflow_from_file(desc_file, overrides=overrides).run()
        output_path = os.path.splitext(desc_file)[0] + f"__{suffix}.csv"
        module_logger.info(f"writing csv: {output_path}")
        result.to_csv(output_path)
