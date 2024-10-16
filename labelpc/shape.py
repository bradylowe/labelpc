import copy
import math
import numpy as np

from qtpy import QtCore
from qtpy import QtGui

import labelpc.utils


# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


R, G, B = SHAPE_COLOR = 0, 255, 0  # green
DEFAULT_LINE_COLOR = QtGui.QColor(R, G, B, 128)                # bf hovering
DEFAULT_FILL_COLOR = QtGui.QColor(R, G, B, 128)                # hovering
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)        # selected
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(R, G, B, 155)         # selected
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(R, G, B, 255)         # hovering
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 255, 255, 255)  # hovering


class Shape(object):

    P_SQUARE, P_ROUND = 0, 1

    MOVE_VERTEX, NEAR_VERTEX = 0, 1

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label=None, line_color=None, shape_type=None,
                 flags=None, group_id=None, orient=None, rack_id=None):
        self.label = label
        self.group_id = group_id
        self.orient = orient
        self.rack_id = rack_id
        self.points = []
        self.lines = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type
        self.flags = flags
        self.crosshairs = False

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.shape_type = shape_type

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = 'polygon'
        if value not in ['polygon', 'rectangle', 'point', 'line', 'circle', 'linestrip']:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def addLine(self, line):
            self.lines.append(line)

    def canAddPoint(self):
        return self.shape_type in ['polygon', 'linestrip']

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def removePoint(self, i):
        self.points.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if self.points:
            color = self.select_line_color \
                if self.selected else self.line_color
            pen = QtGui.QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()

            if self.shape_type == 'rectangle':
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                if self.label == 'pallet':
                    self._vertex_fill_color = self.vertex_fill_color
                else:
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])
            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.crosshairs:
                r, g, b = self.line_color.red(), self.line_color.green(), self.line_color.blue()
                self.line_color = QtGui.QColor(r, g, b, 20)
                for line in self.lines:
                    painter.drawLine(line)
            elif self.label and 'rack' in self.label and self.lines:
                painter.drawLine(self.lines[0])
            if self.fill:
                color = self.select_fill_color \
                    if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        if self.label == 'pallet':
            return
        min_distance = float('inf')
        min_i = None
        for i, p in enumerate(self.points):
            dist = labelpc.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float('inf')
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = labelpc.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type == 'rectangle':
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    @staticmethod
    def rectangleToPolygon(rect):
        polygon = rect.copy()
        polygon.shape_type = 'polygon'
        p1, p2 = rect.points
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        polygon.points = []
        polygon.addPoint(QtCore.QPointF(x1, y1))
        polygon.addPoint(QtCore.QPointF(x1, y2))
        polygon.addPoint(QtCore.QPointF(x2, y2))
        polygon.addPoint(QtCore.QPointF(x2, y1))
        return polygon

    def rectIntersection(self, other):
        xA = max(self.points[0].x(), other.points[0].x())
        yA = min(self.points[0].y(), other.points[0].y())
        xB = min(self.points[1].x(), other.points[1].x())
        yB = max(self.points[1].y(), other.points[1].y())
        return max(0, xB - xA + 1), max(0, yA - yB + 1)

    def copy(self):
        return copy.deepcopy(self)

    def calculateRackExitEdge(self):
        center = (self.points[0] + self.points[1]) / 2.0
        p1 = QtCore.QPoint(int(center.x()), int(center.y()))
        p2 = QtCore.QPoint(int(center.x()), int(center.y()))
        if self.orient == 0:
            p2.setY(int(self.points[1].y()))
        elif self.orient == 1:
            p2.setX(int(self.points[1].x()))
        elif self.orient == 2:
            p2.setY(int(self.points[0].y()))
        elif self.orient == 3:
            p2.setX(int(self.points[0].x()))
        self.lines = [QtCore.QLine(p1, p2)]

    @property
    def displayName(self):
        name = self.label
        if self.group_id is not None:
            name += " (%d)" % self.group_id
        if self.rack_id is not None:
            name += " (%d)" % self.rack_id
        if self.orient is not None:
            name += " (%d)" % self.orient
        return name

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
