import json
import os
import pickle
from datetime import datetime
from keras.callbacks import History
import config
from workflows.utils import sequential_execution
from workflows.load_data.create_loader import create_loader_workflow
from workflows.model_input.create import CreateModelInputWorkflow, PreprocessedTrainingDataSplit, \
    train_test_split_model_input, PreprocessedModelInput
from workflows.pipeline.create_pipeline import create_pipeline_workflow
from workflows.sequential_model.create import create_sequential_model_workflow, create_model_fit_params, get_best_model


def train(description):
    model_name = description['name']
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if 'repeat' in description['modelTraining']:
        print("NOTE: repeating training not implemented yet!")

    # create session folder
    current_training_path = os.path.join(
        config.get_config().dir_training,
        model_name,
        session_id)
    os.makedirs(current_training_path)

    preprocessed_data = run_pipeline_create_model_input(description)
    data = train_test_split_model_input(model_input=preprocessed_data,
                                        description=description['modelInput'])

    model = create_sequential_model_workflow(
        sequential_model_desc=description['sequentialModel'],
        model_compile=description['modelCompile'],
        input_dim=data.X_train.shape[1:])

    # train
    path_best_model_weights = os.path.join(current_training_path, "best_model")
    fit_params = create_model_fit_params(
        data=data,
        model_training_desc=description['modelTraining'],
        path_best_model=path_best_model_weights
    )
    fit_history: History = model.fit(**fit_params)
    # save result
    path_history = os.path.join(current_training_path, "history.pickle")
    with open(path_history, "wb") as f:
        pickle.dump(fit_history, f)
    with open(os.path.join(current_training_path, "description.json"), "w") as f:
        json.dump(description, f, indent=4)
    best_model = get_best_model(path_to_model=path_best_model_weights, model=model)
    best_model.save(os.path.join(current_training_path, "model.h5"))

    if preprocessed_data.scalers:
        with open(os.path.join(current_training_path, "scalers.pickle"), "wb") as f:
            pickle.dump(preprocessed_data.scalers, f)

    # cleanup
    if os.path.isfile(path_best_model_weights):
        os.remove(path_best_model_weights)

    return current_training_path, best_model


def run_pipeline_create_model_input(description, pretrained_scalers = []):
    composed = [
        create_loader_workflow(description['source']).load,
        create_pipeline_workflow(description['pipeline']).execute,
        CreateModelInputWorkflow(description['modelInput'], pretrained_scalers=pretrained_scalers).model_preprocessing
    ]
    data: PreprocessedModelInput = sequential_execution(composed)
    return data
