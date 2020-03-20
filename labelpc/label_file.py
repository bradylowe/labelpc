import json
import os.path as osp

from labelpc import __version__
from labelpc.logger import logger
from labelpc import PY2


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = []
        self.sourcePath = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        keys = [
            'version',
            'sourcePath',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
        ]
        try:
            with open(filename, 'rb' if PY2 else 'r') as f:
                data = json.load(f)
            version = data.get('version')
            if version is None:
                logger.warn(
                    'Loading JSON file ({}) of unknown version'
                    .format(filename)
                )
            elif version.split('.')[0] != __version__.split('.')[0]:
                logger.warn(
                    'This JSON file ({}) may be incompatible with '
                    'current labelpc. version in file: {}, '
                    'current version: {}'.format(
                        filename, version, __version__
                    )
                )

            sourcePath = data['sourcePath']
            flags = data.get('flags') or {}
            shapes = [
                dict(
                    label=s['label'],
                    points=s['points'],
                    shape_type=s.get('shape_type', 'polygon'),
                    flags=s.get('flags', {}),
                    group_id=s.get('group_id'),
                    rack_id=s.get('rack_id'),
                    orient=s.get('orient')
                )
                for s in data['shapes']
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.sourcePath = sourcePath
        self.filename = filename
        self.otherData = otherData

    def save(
        self,
        filename,
        shapes,
        sourcePath,
        otherData=None,
        flags=None,
    ):
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            sourcePath=sourcePath,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, 'wb' if PY2 else 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

    @staticmethod
    def get_source(filename):
        try:
            with open(filename) as f:
                data = json.load(f)
            sourcePath = data['sourcePath']
            if sourcePath:
                return sourcePath
            else:
                return filename
        except Exception as e:
            return filename.replace('.json', '.las')
