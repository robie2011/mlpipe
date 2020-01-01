import argparse
from mlpipe.cli.actions import list_models, describe_model, train_model, test_model
import logging


module_logger = logging.getLogger()


def actionNotImplemented(args):
    module_logger.error("Action not implemented", args.action)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    # train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("files", metavar="FILES", default=[], nargs='*')


    # test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument("files", metavar="FILES", default=[], nargs='*')

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

    action_switcher.get(args.action, actionNotImplemented)(args)

