import argparse
from cli.actions import list_models, describe_model, train_model, test_model
import logging


def actionNotImplemented(args):
    print("Action not implemented", args.action)


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("file", metavar="FILE")


    # test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument("file", metavar="FILE")

    # list
    parser_list = subparsers.add_parser('list')

    # describe
    parser_describe = subparsers.add_parser("describe")
    parser_describe.add_argument("model_session", metavar="MODEL/SESSION_ID")


    args = parser.parse_args()
    action_switcher = {
        "train": train_model,
        "list": list_models,
        "describe": describe_model,
        "test": test_model
    }


    # create a logging format
    logger = logging.getLogger(__name__)
    print(__name__)

    # add the handlers to the logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    action_switcher.get(args.action, actionNotImplemented)(args)

