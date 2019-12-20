from datetime import datetime
from typing import cast
from keras.callbacks import History
from config import TrainingProject
from workflows.load_data.create_loader import create_loader_workflow
from workflows.model_input.create import CreateModelInputWorkflow, train_test_split_model_input, PreprocessedModelInput
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model
from workflows.utils import sequential_execution


def train(description):
    model_name = description['name']
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if 'repeat' in description['modelTraining']:
        print("NOTE: repeating training not implemented yet!")

    with TrainingProject(name=model_name, session_id=session_id) as project:
        path_best_model_weights = project.create_path_tmp_file()

        preprocessed_data = run_pipeline_create_model_input(description)
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


def run_pipeline_create_model_input(description, pretrained_scalers = []):
    composed = [
        create_loader_workflow(description['source']).load,
        create_pipeline_workflow(description['pipeline']).execute,
        CreateModelInputWorkflow(description['modelInput'], pretrained_scalers=pretrained_scalers).model_preprocessing
    ]
    data: PreprocessedModelInput = sequential_execution(composed)
    return data
