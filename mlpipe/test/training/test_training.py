import locale
import unittest

from mlpipe.dsl.interpreter import create_workflow_from_file, create_workflow_from_yaml
from mlpipe.utils.path_tool import get_dir

locale.setlocale(locale.LC_ALL, 'de_CH.UTF-8')


def _get_evaluation_config(model_name: str, session_id: str):
    return f'''
    name: {model_name}
    session: {session_id}
    source:
      name: mlpipe.datasources.empa.EmpaCsvSourceAdapter
      pathToFile: /Users/robert.rajakone/repos/2019_p9/code/mlpipe/test/dsl/testinput.csv
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
    # def test_analyzer(self):
    #     path_to_file = get_dir(["test", "training", "example.training.yml"])
    #     result = create_workflow_from_file(path_to_file)
    #     print(result.package)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
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
            "output": {"name": "mlpipe.integration.output.ConsoleOutput"},
            "executionFrequencyMinutes": 1
        })

        print(manager.run())

    def test_analyze(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        manager = create_workflow_from_file(path_to_file, overrides={"@mode": "analyze"})
        print(manager.run())

    def test_evaluate(self):
        name, session = _get_test_model_name_session()
        config_str = _get_evaluation_config(name, session)
        manager = create_workflow_from_yaml(config_str, {
            "@mode": "evaluate",
        })
        print(manager.run())

    #
    # def test_evaluate_saved_model(self):
    #     main_evaluate_workflow.DISABLE_EVAL_STATS = True
    #     path_to_file = get_dir(["test", "training", "example.training.yml"])
    #     description = FileDescription(path_to_file).load()
    #     result = train(description)
    #     model_path_split = result[0].split("/")
    #     model_name, trained_session_id = model_path_split[-2], model_path_split[-1]
    #
    #     config_str = _get_evaluation_config(model_name=model_name, session_id=trained_session_id)
    #     result = evaluate(YamlStringDescription(config_str).load())
    #
    # def test_evluate_saved_model_real(self):
    #     path_to_file = get_dir_from_code_root(["_descriptions", "test.empa.mlp.simple.yml"])
    #     evaluate(FileDescription(path_to_file).load())