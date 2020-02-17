from mlpipe.workflows.evaluate.binary_classification_evaluator import BinaryClassificationEvaluator
from mlpipe.workflows.evaluate.regression_evaluator import RegressionEvaluator

prediction_evaluators = {
    "binary": BinaryClassificationEvaluator(),
    "regression": RegressionEvaluator()
}
