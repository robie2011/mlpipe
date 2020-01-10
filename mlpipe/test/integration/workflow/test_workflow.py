import unittest
from mlpipe.utils import get_dir_from_code_root
from mlpipe.workflows.main_integration_workflow import IntegrationWorkflow
from mlpipe.workflows.utils import load_description_file


class TestIntegrationWorkflow(unittest.TestCase):
    def test_workflow(self):
        path_test_csv_file = get_dir_from_code_root(["_descriptions", "integrate.empa.simple.yml"])
        description = load_description_file(path_test_csv_file)
        wf = IntegrationWorkflow(description)
        wf.run(limit_execution=1)


if __name__ == '__main__':
    unittest.main()
