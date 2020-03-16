import locale
import unittest

import simplejson

from mlpipe.dsl_interpreter.interpreter import create_workflow_from_file, create_workflow_from_yaml
from mlpipe.utils.path_tool import dir_mlpipe

locale.setlocale(locale.LC_ALL, 'de_CH.UTF-8')


def _get_evaluation_config(model_name: str, session_id: str):
    return f'''
    name: {model_name}
    session: {session_id}
    source:
      name: mlpipe.datasources.empa_csv_source_adapter.EmpaCsvSourceAdapter
      pathToFile: mlpipe/test/dsl/testinput.csv
      fields:
        - abc1 as temp1
        - abc2 as temp2
    '''


_name_session = None


def _get_test_model_name_session():
    global _name_session
    if not _name_session:
        TestWorkflows().test_training()
    return _name_session


class TestWorkflows(unittest.TestCase):
    def setUp(self) -> None:
        self.test_training()

    def test_training(self):
        path_to_file = dir_mlpipe / "test" / "training" / "example.training.yml"
        manager = create_workflow_from_file(path_to_file, overrides={"@mode": "train"})
        train_dir, _model = manager.run()
        print(train_dir, _model)
        global _name_session
        _name_session = train_dir.split("/")[-2], train_dir.split("/")[-1]

    def test_integrate(self):
        name, session = _get_test_model_name_session()
        config_str = _get_evaluation_config(name, session)
        manager = create_workflow_from_yaml(config_str, {
            "limitExecution": 1,
            "@mode": "integrate",
            "integrate": {
                "output": {"name": "mlpipe.integration.output.ConsoleOutput"},
                "executionFrequencyMinutes": 1
            }
        })

        print(manager.run())

    def test_analyze(self):
        path_to_file = dir_mlpipe / "test" / "training" / "example.training.yml"
        manager = create_workflow_from_file(path_to_file, overrides={"@mode": "analyze"})
        result = manager.run()
        data = simplejson.dumps(result, ignore_nan=True, default=lambda o: o.__dict__)
        # write_text_file("/tmp/analytics.json", data)

    def test_evaluate(self):
        name, session = _get_test_model_name_session()
        config_str = _get_evaluation_config(name, session)
        manager = create_workflow_from_yaml(config_str, {
            "@mode": "evaluate",
        })
        print(manager.run())
