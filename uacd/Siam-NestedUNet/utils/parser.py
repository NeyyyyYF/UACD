import argparse as ag
import json
import os

def get_parser_with_args(metadata_json=None):
    parser = ag.ArgumentParser(description='Training change detection network')
    if metadata_json is None:
        metadata_json = os.path.join(os.path.dirname(__file__), '..', 'metadata.json')
    metadata_json = os.path.abspath(metadata_json)

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata

    return None
