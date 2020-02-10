import base64
import json
import os.path as osp

import PIL.Image
from io import BytesIO
from laspy.file import File
import numpy as np

from labelpc import __version__
from labelpc.logger import logger
from labelpc import PY2
from labelpc import QT4
from labelpc import utils
from labelpc.pointcloud.Voxelize import VoxelGrid

PIL.Image.MAX_IMAGE_PIXELS = None


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_point_cloud_file(filename, mesh=0.05, thickness=0.2, max_points=5000000):
        with File(filename) as f:
            points = np.array((f.x, f.y, f.z)).T
            if len(points) > max_points:
                points = points[np.random.choice(max_points, len(points))]

        vg = VoxelGrid(points, (mesh, mesh, thickness))
        bitmaps = vg.bitmap2d(max=2048, axis=2)
        # Stack 3 copies on top of each other to form RGB image (but still black and white at this point)
        for i in range(len(bitmaps)):
            bitmaps[i] = np.dstack((bitmaps[i], bitmaps[i], bitmaps[i]))
        data = []
        for m in bitmaps:
            img = PIL.Image.fromarray(np.asarray(np.clip(m, 0, 255), dtype="uint8"))
            buff = BytesIO()
            img.save(buff, format="JPEG")
            buff.seek(0)
            data.append(buff.read())
        return data, vg.min_corner(), mesh


    def load(self, filename):
        keys = [
            'version',
            'imagePath',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
            'imageHeight',
            'imageWidth',
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
                    'current labelme. version in file: {}, '
                    'current version: {}'.format(
                        filename, version, __version__
                    )
                )

            imagePath = data['imagePath']
            flags = data.get('flags') or {}
            shapes = [
                dict(
                    label=s['label'],
                    points=s['points'],
                    shape_type=s.get('shape_type', 'polygon'),
                    flags=s.get('flags', {}),
                    group_id=s.get('group_id')
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
        self.imagePath = imagePath
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                'imageHeight does not match with imageData or imagePath, '
                'so getting imageHeight from actual image.'
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                'imageWidth does not match with imageData or imagePath, '
                'so getting imageWidth from actual image.'
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
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
            imagePath=imagePath,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
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
