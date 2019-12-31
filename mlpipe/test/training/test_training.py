import io
import unittest
import yaml
from mlpipe.utils import get_dir
from mlpipe.workflows import main_evaluate_workflow
from mlpipe.workflows.analyzers.create_analyzers import create_analyzer_workflow
from mlpipe.workflows.load_data.create_loader import create_loader_workflow
from mlpipe.workflows.main_evaluate_workflow import evaluate
from mlpipe.workflows.main_training_workflow import train
from mlpipe.workflows.pipeline.create_pipeline import create_pipeline_workflow
from mlpipe.workflows.utils import sequential_execution
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
        description = yaml.load(open(path_to_file, "r"))
        desc_src = description['source']
        desc_pipeline = description['pipeline']
        desc_analyze = description['analyze']

        composed = [
            create_loader_workflow(desc_src).load,
            create_pipeline_workflow(desc_pipeline).execute,
            create_analyzer_workflow(desc_analyze).run
        ]

        data = sequential_execution(composed)
        print(data)

    def test_training(self):
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        result = train(description)

    def test_evaluate_saved_model(self):
        main_evaluate_workflow.DISABLE_EVAL_STATS = True
        path_to_file = get_dir(["test", "training", "example.training.yml"])
        description = yaml.load(open(path_to_file, "r"))
        result = train(description)
        model_path_split = result[0].split("/")
        model_name, trained_session_id = model_path_split[-2], model_path_split[-1]

        str_io = io.StringIO(_get_evaluation_config(model_name=model_name, session_id=trained_session_id))
        str_io.seek(0)
        description = yaml.load(str_io)
        result = evaluate(description)
