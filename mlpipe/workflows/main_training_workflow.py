import logging
from datetime import datetime
from keras.callbacks import History
from mlpipe.config.training_project import TrainingProject
from .description_evaluator.evaluator import execute_from_object
from .model_input.create import train_test_split_model_input
from .sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model

module_logger = logging.getLogger(__name__)


def train(description):
    model_name = description['name']
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    with TrainingProject(name=model_name, session_id=session_id, create=True) as project:
        path_best_model_weights = project.create_path_tmp_file()

        execution_result = execute_from_object(description)
        preprocessed_data = execution_result.package
        data = train_test_split_model_input(model_input=preprocessed_data,
                                            description=description['modelInput'])

        model = create_sequential_model_workflow(
            sequential_model_desc=description['sequentialModel'],
            model_compile=description['modelCompile'],
            input_dim=data.X_train.shape[1:])

        fit_params = create_model_fit_params(
            data=data,
            model_training_desc=description['modelTraining'],
            path_best_model=path_best_model_weights
        )

        fit_history: History = model.fit(**fit_params)
        best_model = get_best_model(path_to_model=path_best_model_weights, model=model)

        project.history = fit_history
        project.description = description
        project.model = best_model
        project.scalers = preprocessed_data.scalers

        return project.path_training_dir, best_model
