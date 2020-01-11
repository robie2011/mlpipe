import io
import unittest
import yaml
from mlpipe.utils import get_dir, get_dir_from_code_root
from mlpipe.workflows import main_evaluate_workflow
from mlpipe.workflows.analyzers.main_analyze_workflow import create_analyzer_workflow
from mlpipe.workflows.description_evaluator import FileDescription, YamlStringDescription
from mlpipe.workflows.load_data.create_loader import create_loader_workflow
from mlpipe.workflows.main_evaluate_workflow import evaluate
from mlpipe.workflows.main_training_workflow import train
from mlpipe.workflows.pipeline.create_pipeline import create_pipeline_workflow
from mlpipe.workflows.description_evaluator.evaluator import sequential_execution, execute_from_file
import locale

locale.setlocale(locale.LC_ALL, 'de_CH.UTF-8')


def _get_evaluation_config(model_name: str, session_id: str):
    return f'''
    name: {model_name}
    session: {session_id}
    testSource:
      name: mlpipe.datasources.empa.EmpaCsvSourceAdapter
      pathToFile: ./mlpipe/test/dsl/testinput.csv
      fields:
        - abc1 as temp1
        - abc2 as temp2
    '''


class TestTraning(unittest.TestCase):
    def test_analyzer(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        result = execute_from_file(path_to_file)
        print(result.package)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = FileDescription(path_to_file).load()
        result = train(description)

    def test_evaluate_saved_model(self):
        main_evaluate_workflow.DISABLE_EVAL_STATS = True
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = FileDescription(path_to_file).load()
        result = train(description)
        model_path_split = result[0].split("/")
        model_name, trained_session_id = model_path_split[-2], model_path_split[-1]

        config_str = _get_evaluation_config(model_name=model_name, session_id=trained_session_id)
        result = evaluate(YamlStringDescription(config_str).load())

    def test_evluate_saved_model_real(self):
        path_to_file = get_dir_from_code_root(["_descriptions", "test.empa.mlp.simple.yml"])
        evaluate(FileDescription(path_to_file).load())