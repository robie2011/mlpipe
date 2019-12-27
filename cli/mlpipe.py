import argparse
from .actions import list_models, describe_model, train_model
import logging


logging.basicConfig(level=logging.DEBUG)


def actionNotImplemented(args):
    raise ValueError("Action not implemented", args.action)


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("file", metavar="FILE")

    parser_test = subparsers.add_parser('test')
    parser_list = subparsers.add_parser('list')

    parser_describe = subparsers.add_parser("describe")
    parser_describe.add_argument("model_session", metavar="MODEL/SESSION_ID")


    args = parser.parse_args()

    action_switcher = {
        "train": train_model,
        "list": list_models,
        "describe": describe_model
    }

    action_switcher.get(args.action, actionNotImplemented)(args)

