# -*- coding: utf-8 -*-

import functools
import os
import os.path as osp
import re
import webbrowser
from tqdm import tqdm, trange

import PIL.Image
from io import BytesIO
import numpy as np

import imgviz
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets
from PyQt5.QtWidgets import QInputDialog, QDialog, QLineEdit, QDialogButtonBox, QFormLayout, QProgressBar

from labelpc import __appname__
from labelpc import PY2
from labelpc import QT5

from labelpc.dialogs.open_file_dialog import OpenFileDialog

from . import utils
from labelpc.config import get_config
from labelpc.label_file import LabelFile
from labelpc.label_file import LabelFileError
from labelpc.logger import logger
from labelpc.shape import Shape
from labelpc.widgets import Canvas
from labelpc.widgets import ColorDialog
from labelpc.widgets import LabelDialog
from labelpc.widgets import LabelQListWidget
from labelpc.widgets import ToolBar
from labelpc.widgets import UniqueLabelQListWidget
from labelpc.widgets import ZoomWidget

from labelpc.pointcloud.PointCloud import PointCloud
from labelpc.pointcloud.Voxelize import VoxelGrid


# TODO:
#   --- BYU students:
#   Snap to corner
#   Snap to center
#   --- Brady:
#   Create annotations for individual slices ??? (floor, lights, evap. coils)
#   Interpolate beam positions inside wall bounds or canvas bounds
#   Make different shapes for i-beam and square beam
#   --- Austin:
#   //DONE Add distance threshold for snap functions to config file (snapToCenter, snapToCorner, rackSep, rackSplit)
#   //DONE Draw crosshairs on beams that span the canvas (toggle on/off)
#   Color one side of rectangle a different color based on group ID
#   //DONE Toggle individual annotations on/off (turn off SHOWALL)
#   Create icons for buttons
#   Create shortcuts

LABEL_COLORMAP = imgviz.label_colormap(value=200)


class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):

        if output is not None:
            logger.warning(
                'argument output is deprecated, use output_file instead'
            )
            if output_file is None:
                output_file = output

        # see labelpc/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config['labels'],
            sort_labels=self._config['sort_labels'],
            show_text_field=self._config['show_label_text_field'],
            completion=self._config['label_completion'],
            fit_to_content=self._config['fit_to_content'],
            flags=self._config['label_flags']
        )

        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr('Flags'), self)
        self.flag_dock.setObjectName('Flags')
        self.flag_widget = QtWidgets.QListWidget()
        if config['flags']:
            self.loadFlags({k: False for k in config['flags']})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList = LabelQListWidget()
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.toggleIndivPolygon)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.setParent(self)
        self.shape_dock = QtWidgets.QDockWidget(
            self.tr('Polygon Labels'),
            self
        )
        self.shape_dock.setObjectName('Labels')
        self.shape_dock.setWidget(self.labelList)
        #self.shape_dock.itemChanged.connect(self.toggleIndivPolygon)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.itemActivated.connect(self.modeSelectionChanged)
        self.uniqLabelList.itemSelectionChanged.connect(self.modeSelectionChanged)
        self.uniqLabelList.setToolTip(self.tr(
            "Select label to start annotating for it. "
            "Press 'Esc' to deselect."))
        if self._config['labels']:
            for label in self._config['labels']:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr(u'Label List'), self)
        self.label_dock.setObjectName(u'Label List')
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr('Search Filename'))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged
        )
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr(u'File List'), self)
        self.file_dock.setObjectName(u'Files')
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config['epsilon'],
            double_click=self._config['canvas']['double_click'],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.nextSliceRequest.connect(self.showNextSlice)
        self.canvas.lastSliceRequest.connect(self.showLastSlice)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.breakRack.connect(self.breakRackManual)
        self.canvas.rackChanged.connect(self.finalizeRack)
        self.canvas.beamChanged.connect(self.finalizeBeam)
        self.canvas.rotateRack.connect(self.rotateRack)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ['flag_dock', 'label_dock', 'shape_dock', 'file_dock']:
            if self._config[dock]['closable']:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]['floatable']:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]['movable']:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]['show'] is False:
                getattr(self, dock).setInvisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config['shortcuts']
        quit = action(self.tr('&Quit'), self.close, shortcuts['quit'], 'quit',
                      self.tr('Quit application'))
        open_ = action(self.tr('&Open'),
                       self.openPointCloud,
                       shortcuts['open'],
                       'open',
                       self.tr('Open point cloud file'))
        opendir = action(self.tr('&Open Dir'), self.openDirDialog,
                         shortcuts['open_dir'], 'open', self.tr(u'Open Dir'))
        showNextSlice = action(
            self.tr('Next Slice'),
            self.showNextSlice,
            shortcuts['open_next'],
            'next slice',
            self.tr(u'Show next slice of point cloud'),
            enabled=False,
        )
        showLastSlice = action(
            self.tr('Last Slice'),
            self.showLastSlice,
            shortcuts['open_prev'],
            'next slice',
            self.tr(u'Show previous slice of point cloud'),
            enabled=False,
        )
        highlight_slice = action(self.tr('Highlight Slice'), self.highlightSlice, shortcuts['highlight_slice'],
                                 'highlight', self.tr('Highlight the currently showing slice of the point cloud'),

                                 enabled=False)

        check_highlight_slice = action(self.tr('Highlight Slice'), self.checkHighlightSlice, shortcuts['highlight_slice'],
                                 'eye', self.tr('Highlight the currently showing slice of the point cloud'),
                                 checkable=True,
                                 enabled=False)

        save = action(self.tr('&Save'),
                      self.saveFile, shortcuts['save'], 'save',
                      self.tr('Save labels to file'), enabled=False)
        saveAs = action(self.tr('&Save As'), self.saveFileAs,
                        shortcuts['save_as'],
                        'save-as', self.tr('Save labels to a different file'),
                        enabled=False)

        deleteFile = action(
            self.tr('&Delete File'),
            self.deleteFile,
            shortcuts['delete_file'],
            'delete',
            self.tr('Delete current label file'),
            enabled=False)

        changeOutputDir = action(
            self.tr('&Change Output Dir'),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts['save_to'],
            icon='open',
            tip=self.tr(u'Change where annotations are loaded/saved')
        )

        saveAuto = action(
            text=self.tr('Save &Automatically'),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon='save',
            tip=self.tr('Save automatically'),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config['auto_save'])

        saveWithImageData = action(
            text='Save With Image Data',
            slot=self.enableSaveImageWithData,
            tip='Save image data in label file',
            checkable=True,
            checked=False,
            #checked=self._config['store_data'],
        )

        close = action('&Close', self.closeFile, shortcuts['close'], 'close',
                       'Close current file')

        render_3d = action('Render points in 3D', self.render3d, None, 'render', 'Render the points in 3D')

        highlight_walls = action('Highlight walls', self.highlightWalls, None, 'highlight', 'Highlight walls')

        view_annotation_3d = action('View Label 3D', self.viewAnnotation3d, None, 'view 3d', 'View annotation 3d')

        update_annotation = action('Update Label', self.updateSelectedLabelWithHighlightedPoints, None, 'update label',
                                   'Update the label based on the points currently highlighted in the 3d viewer')

        break_all_racks = action('Break All Racks', self.breakAllRacks, None, None,
                                 'Break the racks that are broken up due to proximity to support beams')

        merge_racks = action('Merge Racks', self.unbreakRack, None, None,
                             'Merge the selected racks into a single rack (undo rack break)')

        show_crosshairs = action('Show beam crosshairs', self.showCrosshairs, None, 'eye',
                                 'show crosshairs over beam annotations', checkable=True)

        toggle_racks = action('Toggle racks', self.toggleRacks, None, 'eye', 'Toggle rack rendering on/off')

        toggle_pallets = action('Toggle pallets', self.togglePallets, None, 'eye', 'Toggle pallet rendering on/off')

        toggle_walls = action('Toggle walls', self.toggleWalls, None, 'eye', 'Toggle walls rendering on/off')

        toggle_beams = action('Toggle beams', self.toggleBeams, None, 'eye', 'Toggle beam rendering on/off')

        rotate_rack = action('Rotate Rack', self.rotateRack, None, None,
                             'Change the orientation of the selected rack')

        predict_pallets = action('Predict pallets', self.predictPalletsForAllRacks, None, None,
                                 'Predict the pallet locations for all the rack annotations')

        select_pallets_by_rack = action('Select pallets by rack', self.selectPalletsByRack, None, None,
                                        'Select all pallets that belong to the selected rack')

        select_pallets_by_group = action('Select pallets by group', self.selectPalletsByGroup, None, None,
                                         'Select all pallets that belong to the selected rack group')

        select_beams = action('Select beams', self.selectBeams, None, 'select beams', 'Select all beams')

        select_beam_row = action('Select beam row', self.selectBeamRow, None, None,
                                 'Select all the beams in the selected row')

        select_beam_column = action('Select beam column', self.selectBeamColumn, None, None,
                                    'Select all the beams in the selected column')

        select_pallets = action('Select pallets', self.selectPallets, None, None, 'Select all pallets')

        interpolate_beams = action('Interp beams', self.interpolateBeamPositions, None, None,
                                   'Interpolate beam positions based on existing beam positions')

        user_tighten_rack = action('Tighten rack', self.userTightenRack, None, None,
                                   'Tighten the selected rack around the points in the point cloud')

        annotate_3d_beam = action('Annotate 3D beam', self.annotate3dBeam, None, None,
                                  'Use the points highlighted in the 3D viewer to create beam annotation')

        convert_to_from_i_beam = action('Convert to/from I-beam', self.convertToFromIBeam, None, None,
                                        'Convert the selected beams from one beam type to another')

        toggle_keep_prev_mode = action(
            self.tr('Keep Previous Annotation'),
            self.toggleKeepPrevMode,
            shortcuts['toggle_keep_prev_mode'], None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True)
        toggle_keep_prev_mode.setChecked(self._config['keep_prev'])

        createMode = action(
            self.tr('Create Polygons'),
            lambda: self.toggleDrawMode(False, createMode='polygon'),
            shortcuts['create_polygon'],
            'objects',
            self.tr('Start drawing polygons'),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr('Create Rectangle'),
            lambda: self.toggleDrawMode(False, createMode='rectangle'),
            shortcuts['create_rectangle'],
            'objects',
            self.tr('Start drawing rectangles'),
            enabled=False,
        )
        createCircleMode = action(
            self.tr('Create Circle'),
            lambda: self.toggleDrawMode(False, createMode='circle'),
            shortcuts['create_circle'],
            'objects',
            self.tr('Start drawing circles'),
            enabled=False,
        )
        createLineMode = action(
            self.tr('Create Line'),
            lambda: self.toggleDrawMode(False, createMode='line'),
            shortcuts['create_line'],
            'objects',
            self.tr('Start drawing lines'),
            enabled=False,
        )
        createPointMode = action(
            self.tr('Create Point'),
            lambda: self.toggleDrawMode(False, createMode='point'),
            shortcuts['create_point'],
            'objects',
            self.tr('Start drawing points'),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr('Create LineStrip'),
            lambda: self.toggleDrawMode(False, createMode='linestrip'),
            shortcuts['create_linestrip'],
            'objects',
            self.tr('Start drawing linestrip. Ctrl+LeftClick ends creation.'),
            enabled=False,
        )
        editMode = action(self.tr('Edit Polygons'), self.setEditMode,
                          shortcuts['edit_polygon'], 'edit',
                          self.tr('Move and edit the selected polygons'),
                          enabled=False)

        delete = action(self.tr('Delete Polygons'), self.deleteSelectedShape,
                        shortcuts['delete_polygon'], 'cancel',
                        self.tr('Delete the selected polygons'), enabled=False)

        copy = action(self.tr('Duplicate Polygons'), self.copySelectedShape,
                      shortcuts['duplicate_polygon'], 'copy',
                      self.tr('Create a duplicate of the selected polygons'),
                      enabled=False)
        undoLastPoint = action(self.tr('Undo last point'),
                               self.canvas.undoLastPoint,
                               shortcuts['undo_last_point'], 'undo',
                               self.tr('Undo last drawn point'), enabled=False)
        addPointToEdge = action(
            self.tr('Add Point to Edge'),
            self.canvas.addPointToEdge,
            None,
            'edit',
            self.tr('Add point to the nearest edge'),
            enabled=False,
        )
        removePoint = action(
            text='Remove Selected Point',
            slot=self.canvas.removeSelectedPoint,
            icon='edit',
            tip='Remove selected point from polygon',
            enabled=False,
        )

        undo = action(self.tr('Undo'), self.undoShapeEdit,
                      shortcuts['undo'], 'undo',
                      self.tr('Undo last add and edit of shape'),
                      enabled=False)

        hideAll = action(self.tr('&Hide\nPolygons'),
                         functools.partial(self.togglePolygons, False),
                         icon='eye', tip=self.tr('Hide all polygons'),
                         enabled=False)
        showAll = action(self.tr('&Show\nPolygons'),
                         functools.partial(self.togglePolygons, True),
                         icon='eye', tip=self.tr('Show all polygons'),
                         enabled=False)

        help = action(self.tr('&Tutorial'), self.tutorial, icon='help',
                      tip=self.tr('Show tutorial page'))

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            self.tr(
                'Zoom in or out of the image. Also accessible with '
                '{} and {} from the canvas.'
            ).format(
                utils.fmtShortcut(
                    '{},{}'.format(
                        shortcuts['zoom_in'], shortcuts['zoom_out']
                    )
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(self.tr('Zoom &In'),
                        functools.partial(self.addZoom, 1.1),
                        shortcuts['zoom_in'], 'zoom-in',
                        self.tr('Increase zoom level'), enabled=False)
        zoomOut = action(self.tr('&Zoom Out'),
                         functools.partial(self.addZoom, 0.9),
                         shortcuts['zoom_out'], 'zoom-out',
                         self.tr('Decrease zoom level'), enabled=False)
        zoomOrg = action(self.tr('&Original size'),
                         functools.partial(self.setZoom, 100),
                         shortcuts['zoom_to_original'], 'zoom',
                         self.tr('Zoom to original size'), enabled=False)
        fitWindow = action(self.tr('&Fit Window'), self.setFitWindow,
                           shortcuts['fit_window'], 'fit-window',
                           self.tr('Zoom follows window size'), checkable=True,
                           enabled=False)
        fitWidth = action(self.tr('Fit &Width'), self.setFitWidth,
                          shortcuts['fit_width'], 'fit-width',
                          self.tr('Zoom follows window width'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg,
                       fitWindow, fitWidth)
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(self.tr('&Edit Label'), self.editLabel,
                      shortcuts['edit_label'], 'edit',
                      self.tr('Modify the label of the selected polygon'),
                      enabled=False)

        fill_drawing = action(
            self.tr('Fill Drawing Polygon'),
            self.canvas.setFillDrawing,
            None,
            'color',
            self.tr('Fill polygon while drawing'),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save, saveAs=saveAs, open=open_, close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete, edit=edit, copy=copy,
            undoLastPoint=undoLastPoint, undo=undo,
            addPointToEdge=addPointToEdge, removePoint=removePoint,
            createMode=createMode, editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
            fitWindow=fitWindow, fitWidth=fitWidth,
            zoomActions=zoomActions,
            showNextSlice=showNextSlice, showLastSlice=showLastSlice,
            highlightSlice=highlight_slice,
            checkHighlightSlice=check_highlight_slice,
            render3d=render_3d,
            showCrosshairs=show_crosshairs,
            toggleRacks=toggle_racks, togglePallets=toggle_pallets, toggleWalls=toggle_walls,
            toggleBeams=toggle_beams,
            highlightWalls=highlight_walls,
            viewAnnotation3d=view_annotation_3d,
            updateAnnotation=update_annotation,
            breakAllRacks=break_all_racks,
            mergeRacks=merge_racks,
            rotateRack=rotate_rack,
            predictPallets=predict_pallets,
            selectPalletsByRack=select_pallets_by_rack,
            selectPalletsByGroup=select_pallets_by_group,
            selectBeamColumn=select_beam_column,
            selectBeamRow=select_beam_row,
            interpolateBeams=interpolate_beams,
            userTightenRack=user_tighten_rack,
            annotate3dbeam=annotate_3d_beam,
            convertToFromIBeam=convert_to_from_i_beam,
            #fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            fileMenuActions=(open_, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                copy,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                addPointToEdge,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                None,
                editMode,
                edit,
                copy,
                delete,
                undo,
                undoLastPoint,
                addPointToEdge,
                removePoint,
                None,
                select_pallets_by_rack,
                select_pallets_by_group,
                select_beams, select_beam_column, select_beam_row,
                select_pallets,
                user_tighten_rack,
            ),
            onLoadActive=(
                close,
                showNextSlice,
                showLastSlice,
                render_3d,
                highlight_walls,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, rotate_rack, merge_racks, break_all_racks, update_annotation,
                             predict_pallets, select_pallets_by_rack, select_pallets_by_group, interpolate_beams),
            on3dViewerActive=(
                highlight_slice,
                check_highlight_slice,
            ),
        )

        self.canvas.edgeSelected.connect(self.canvasShapeEdgeSelected)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr('&File')),
            edit=self.menu(self.tr('&Edit')),
            view=self.menu(self.tr('&View')),
            help=self.menu(self.tr('&Help')),
            recentFiles=QtWidgets.QMenu(self.tr('Open &Recent')),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                showNextSlice,
                showLastSlice,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                render_3d,
                view_annotation_3d,
                check_highlight_slice,
                None,
                show_crosshairs,
                toggle_racks,
                toggle_pallets,
                toggle_walls,
                toggle_beams,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                None,
                fitWindow,
                fitWidth,
                None,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action('&Copy here', self.copyShape),
                action('&Move here', self.moveShape),
            ),
        )

        self.tools = self.toolbar('Tools')
        # Menu buttons on Left
        self.actions.tool = (
            open_,
            showNextSlice,
            showLastSlice,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            copy,
            delete,
            undo,
            None,
            zoomIn,
            zoom,
            zoomOut,
            fitWindow,
            fitWidth,
            render_3d,
            highlight_walls,
            update_annotation,
            break_all_racks,
            merge_racks,
            rotate_rack,
            predict_pallets,
            select_pallets_by_group,
            select_pallets_by_rack,
            interpolate_beams,
            annotate_3d_beam,
            convert_to_from_i_beam,
        )

        self.statusBar().showMessage(self.tr('%s started.') % __appname__)
        self.statusBar().show()
        self.progressBar = QProgressBar()


        self.statusBar().addPermanentWidget(self.progressBar)

        # This is simply to show the bar
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.setMaximum(100)
        #self.progressBar.setValue(50)

        if output_file is not None and self._config['auto_save']:
            logger.warn(
                'If `auto_save` argument is True, `output_file` argument '
                'is ignored and output filename is automatically '
                'set as IMAGE_BASENAME.json.'
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.sourcePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.max_points = None
        self.scale = None
        self.thickness = None
        self.offset = None
        self.annotationMode = None
        self.imageData = None
        self.sliceIndices = None
        self.renderingRacks, self.renderingWalls, self.renderingPallets = True, True, True
        self.renderingBeams = True,
        self.pointcloud = PointCloud(render=False)
        self._cur_group = 0
        self._cur_rack = 0
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config['file_search']:
            self.fileSearch.setText(config['file_search'])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings('labelpc', 'labelpc')
        # FIXME: QSettings.value can return None on PyQt4
        self.recentFiles = self.settings.value('recentFiles', []) or []
        size = self.settings.value('window/size', QtCore.QSize(600, 500))
        position = self.settings.value('window/position', QtCore.QPoint(0, 0))
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(
            self.settings.value('window/state', QtCore.QByteArray()))

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not self.labelList.itemsToShapes

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        if self._config['auto_save'] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.sourcePath)[0] + '.json'
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}*'.format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = '{} - {}'.format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, viewer=None, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)
        if viewer is not None:
            for action in self.actions.on3dViewerActive:
                action.setEnabled(viewer)

    def canvasShapeEdgeSelected(self, selected, shape):
        self.actions.addPointToEdge.setEnabled(
            selected and shape and shape.canAddPoint()
        )

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.image = QtGui.QImage()
        self.labelList.clear()
        self.sliceIdx = 0
        self.filename = None
        self.sourcePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.max_points = None
        self.thickness = None
        self.scale = None
        self.offset = None
        self.annotationMode = None
        self.sliceIndices = None
        self.imageData = None
        self.renderingRacks, self.renderingWalls, self.renderingPallets = True, True, True
        self.renderingBeams = True,
        self.pointcloud.close_viewer()
        self.pointcloud = PointCloud(render=False)
        self._cur_group = 0
        self._cur_rack = 0
        self.canvas.resetState()
        self.highlightSliceOnScroll = False

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = 'https://github.com/wkentaro/labelme/tree/master/examples/tutorial'  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode='polygon', showPopup=True):
        self._config['display_label_popup'] = showPopup
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode

        if createMode not in ['polygon', 'rectangle', 'circle', 'line', 'point', 'linestrip']:
            raise ValueError('Unsupported createMode: %s' % createMode)

        self.actions.createMode.setEnabled(createMode != 'polygon')
        self.actions.createRectangleMode.setEnabled(createMode != 'rectangle')
        self.actions.createCircleMode.setEnabled(createMode != 'circle')
        self.actions.createLineMode.setEnabled(createMode != 'line')
        self.actions.createPointMode.setEnabled(createMode != 'point')
        self.actions.createLineStripMode.setEnabled(createMode != 'linestrip')
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon('labels')
            action = QtWidgets.QAction(
                icon, '&%d %s' % (i + 1, QtCore.QFileInfo(f).fileName()), self)
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config['validate_label'] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config['validate_label'] in ['exact']:
                if label_i == label:
                    return True
        return False

    def editLabel(self, item=False):
        if item and not isinstance(item, QtWidgets.QListWidgetItem):
            raise TypeError('unsupported type of item: {}'.format(type(item)))

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = self.labelList.get_shape_from_item(item)
        if shape is None:
            return
        text, flags, group_id = self.labelDialog.popUp(
            text=shape.label, flags=shape.flags, group_id=shape.group_id,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr('Invalid label'),
                self.tr(
                    "Invalid label '{}' with validation type '{}'"
                ).format(text, self._config['validate_label'])
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        item.setText(shape.displayName)
        self.setDirty()
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(role=Qt.UserRole, value=shape.label)
            self.uniqLabelList.addItem(item)

    def modeSelectionChanged(self):
        items = self.uniqLabelList.selectedItems()
        if not items:
            self._config['display_label_popup'] = True
            self.annotationMode = None
            return
        label = items[0].data(Qt.UserRole)
        if label not in self._config['labels']:
            self._config['display_label_popup'] = True
            self.annotationMode = None
            return

        self.annotationMode = label
        if label in ['pole', 'beam', 'I_beam']:
            self.toggleDrawMode(False, createMode='point', showPopup=False)
        elif 'rack' in label or label == 'noise':
            self.toggleDrawMode(False, createMode='rectangle', showPopup=False)
        elif label == 'door':
            self.toggleDrawMode(False, createMode='line')
        elif label == 'walls':
            self.toggleDrawMode(False, createMode='polygon', showPopup=False)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.get_item_from_shape(shape)
            item.setSelected(True)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        if n_selected == 1 and self.pointcloud.viewer_is_ready():
            self.highlightPointsInLabel(self.canvas.selectedShapes[0])

    def addLabel(self, shape):
        text = shape.displayName
        item = QtWidgets.QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.labelList.itemsToShapes.append((item, shape))
        self.labelList.addItem(item)
        qlabel = QtWidgets.QLabel()
        qlabel.setText(text)
        qlabel.setAlignment(QtCore.Qt.AlignBottom)
        item.setSizeHint(qlabel.sizeHint())
        self.labelList.setItemWidget(item, qlabel)
        if not self.uniqLabelList.findItemsByLabel(shape.label):
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        rgb = self._get_rgb_by_label(shape.label)
        if rgb is None:
            return

        r, g, b = rgb
        qlabel.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'
            .format(text, r, g, b)
        )
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        #shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.fill_color = QtGui.QColor(r, g, b, 64)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        #shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        shape.select_fill_color = QtGui.QColor(r, g, b, 128)

    def _get_rgb_by_label(self, label):
        if self._config['shape_color'] == 'auto':
            item = self.uniqLabelList.findItemsByLabel(label)[0]
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config['shift_auto_shape_color']
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (self._config['shape_color'] == 'manual' and
              self._config['label_colors'] and
              label in self._config['label_colors']):
            return self._config['label_colors'][label]
        elif self._config['default_shape_color']:
            return self._config['default_shape_color']

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.get_item_from_shape(shape)
            self.labelList.takeItem(self.labelList.row(item))
            idx = self.labelList.get_index_from_shape(shape)
            if idx is not None:
                del self.labelList.itemsToShapes[idx]
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            shape_type = shape['shape_type']
            flags = shape['flags']
            group_id = shape.get('group_id')
            rack_id = shape.get('rack_id')
            orient = shape.get('orient')

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                rack_id=rack_id,
                orient=orient
            )
            if shape.group_id is not None:
                self._cur_group = max(self._cur_group, shape.group_id)
            if shape.rack_id is not None:
                self._cur_rack = max(self._cur_rack, shape.rack_id)
            for p in points:
                shape.addPoint(self.pointcloudToQpoint(p))
            shape.close()
            if 'beam' in shape.label:
                shape.lines = self.canvas.getEdges(shape)
                if shape.label == 'I_beam':
                    shape.point_type = Shape.P_SQUARE
            elif 'rack' in shape.label:
                shape.calculateRackExitEdge()

            default_flags = {}
            if self._config['label_flags']:
                for pattern, keys in self._config['label_flags'].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            # Todo: figure out if we need flags or not
            #shape.flags.update(flags)

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            return dict(
                label=s.label.encode('utf-8') if PY2 else s.label,
                points=[self.qpointToPointcloud(p) for p in s.points],
                group_id=s.group_id,
                rack_id=s.rack_id,
                orient=s.orient,
                shape_type=s.shape_type,
                flags=s.flags
            )

        shapes = [format_shape(shape) for shape in self.labelList.shapes]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                sourcePath=self.sourcePath,
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                osp.dirname(self.sourcePath), Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError('There are duplicate files.')
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr('Error saving label data'),
                self.tr('<b>%s</b>') % e
            )
            return False

    def copySelectedShape(self):
        added_shapes = self.canvas.copySelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                shape = self.labelList.get_shape_from_item(item)
                selected_shapes.append(shape)
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = self.labelList.get_shape_from_item(item)
        self.canvas.setShapeVisible(shape)

    # Callback functions:

    def viewAnnotation3d(self):
        items = self.labelList.selectedItems()
        if items:
            shape = self.labelList.get_shape_from_item(items[0])
            transformed = []
            for p in shape.points:
                transformed.append(self.qpointToPointcloud(p))
            lookat = np.average(transformed, axis=0)
            self.viewLocation3d(lookat)
            show = self.pointsInShape(shape)
            self.pointcloud.render(self.pointcloud.select(show, highlighted=False))

    def viewLocation3d(self, location):
        if not self.pointcloud.viewer_is_ready():
            self.render3d()
        if len(location) < 3:
            location = np.array((location[0], location[1], 3.0))
        self.pointcloud.viewer.set(lookat=location, theta=np.pi/4., r=15.0, phi=-np.pi/2.)

    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        # Get the label name from the uniqLabelList selected items
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None

        # Get label name and group id from user in popup window
        if self._config['display_label_popup'] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr('Invalid label'),
                self.tr(
                    "Invalid label '{}' with validation type '{}'"
                ).format(text, self._config['validate_label'])
            )
            text = ''
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            # If this is a new pole or beam, snap the annotation to the center of the object
            if 'beam' in text:
                self.finalizeBeam(shape)
            elif text == 'pole':
                transformed = self.qpointToPointcloud(shape.points[0])
                snapped = self.pointcloud.snap_to_center(transformed, self._config['snap_center_thresh'])
                if not np.any(np.isnan(snapped)):
                    shape.points[0] = self.pointcloudToQpoint(snapped)
            # If this is a new wall, snap the points to the corners of the walls
            elif text in ['wall', 'walls']:
                for i, p in enumerate(shape.points):
                    transformed = self.qpointToPointcloud(p)
                    snapped = self.pointcloud.snap_to_corner(transformed, self._config['snap_corner_thresh'])
                    if not np.any(np.isnan(snapped)):
                        shape.points[i] = self.pointcloudToQpoint(snapped)
            # If this is a new rack, split the rack into two racks if necessary and tighten box(es) to rack
            # Calculate the orientation of the rack, and resolve any collisions with other racks, walls, or noise
            # Calculate which rack group these racks belong to and generate a rack_id for each new rack
            elif 'rack' in text:
                shape.rack_id = self.nextRackId()
                self.tightenRack(shape)
                shape.orient = self.rackOrientation(shape)
                #self.showRackHistogram(shape, axis='short')
                #self.showRackHistogram(shape, axis='long')
                if self.isTwoRacks(shape):
                    shape.group_id = self.nextGroupId()
                    self.breakBackToBackRacks(shape)
                else:
                    shape.group_id = self.calculateRackGroupId(shape)
                    self.normalizeRackDimensions(shape)
                    self.resolveRackRectIntersection(shape)
                    self.resolveRackRectIntersection(shape, noise=True)
                    self.resolveRackWallIntersection(shape)
                    shape.calculateRackExitEdge()
                    if not self.isRackBigEnough(shape):
                        self.remLabels([shape])
            self.addLabel(shape)
            self.updatePixmap()
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
            if self.pointcloud.viewer_is_ready():
                highlight = self.pointsInShape(shape)
                self.pointcloud.highlight(self.pointcloud.select(highlight, highlighted=False))
                self.viewLocation3d(self.qpointToPointcloud(shape.points[0]))
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def beamBreaksRack(self, beam, rack):
        front_dist, back_dist = 2.0, 0.2
        if not rack.orient % 2 and rack.points[0].x() < beam.points[0].x() < rack.points[1].x():
            if rack.orient == 0:
                max_y, min_y = rack.points[0].y() + back_dist / self.scale, rack.points[1].y() - front_dist / self.scale
            else:
                max_y, min_y = rack.points[0].y() + front_dist / self.scale, rack.points[1].y() - back_dist / self.scale
            if min_y < beam.points[0].y() < max_y:
                return True, beam.points[0].x(), 0
        elif rack.orient % 2 and rack.points[1].y() < beam.points[0].y() < rack.points[0].y():
            if rack.orient == 1:
                min_x, max_x = rack.points[0].x() - back_dist / self.scale, rack.points[1].x() + front_dist / self.scale
            else:
                min_x, max_x = rack.points[0].x() - front_dist / self.scale, rack.points[1].x() + back_dist / self.scale
            if min_x < beam.points[0].x() < max_x:
                return True, beam.points[0].y(), 1
        return False, None, None

    def rackOrientation(self, rack):
        # If rack is near canvas edge
        center = (rack.points[0] + rack.points[1]) / 2.0
        if center.x() - 4.0 / self.scale < 0.0:
            return 1
        elif center.y() + 4.0 / self.scale > self.canvas.height():
            return 0
        elif center.x() + 4.0 / self.scale > self.canvas.width():
            return 3
        elif center.y() - 4.0 / self.scale < 0.0:
            return 2
        box = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        walls = self.walls
        # If rack is longer in the x-direction
        if abs(box[0][0] - box[1][0]) > abs(box[0][1] - box[1][1]):
            # Grab a box above and below the rack
            box_up = box + np.array((0.0, self._config[rack.label][1]))
            box_down = box - np.array((0.0, self._config[rack.label][1]))
            if walls is not None:
                center_up = self.pointcloudToQpoint((box_up[0] + box_up[1]) / 2.0)
                center_down = self.pointcloudToQpoint((box_down[0] + box_down[1]) / 2.0)
                # Check to see if the upper or lower box goes outside the walls
                if not walls.containsPoint(center_up):
                    return 2
                elif not walls.containsPoint(center_down):
                    return 0
            # Check to see if there is a bunch of stuff in the upper or lower box
            if self.pointcloud.in_box_2d(box_up).sum() > self.pointcloud.in_box_2d(box_down).sum():
                return 0
            else:
                return 2
        else:
            # Grab a box to the left and right of the rack
            box_left = box + np.array((self._config[rack.label][1], 0.0))
            box_right = box - np.array((self._config[rack.label][1], 0.0))
            if walls is not None:
                center_left = self.pointcloudToQpoint((box_left[0] + box_left[1]) / 2.0)
                center_right = self.pointcloudToQpoint((box_right[0] + box_right[1]) / 2.0)
                # Check to see if the left or right box goes outside of the walls
                if not walls.containsPoint(center_left):
                    return 1
                elif not walls.containsPoint(center_right):
                    return 3
            # Check to see if there is a bunch of stuff in the left or right box
            if self.pointcloud.in_box_2d(box_left).sum() > self.pointcloud.in_box_2d(box_right).sum():
                return 1
            else:
                return 3

    def nextGroupId(self):
        self._cur_group += 1
        return self._cur_group

    def nextRackId(self):
        self._cur_rack += 1
        return self._cur_rack

    def calculateRackGroupId(self, rack):
        """
        Loop over all the existing racks. If we find a rack that is right next to this rack, copy its group_id.
        Otherwise, just return an unused group_id.
        """
        for other in self.racks:
            if rack.orient % 2 != other.orient % 2:
                continue
            center = (other.points[0] + other.points[1]) / 2.0
            if rack.orient % 2:
                offset = QtCore.QPointF(other.points[1].x() - other.points[0].x(), 0.0)
            else:
                offset = QtCore.QPointF(0.0, other.points[1].y() - other.points[0].y())
            if rack.containsPoint(center + offset) or rack.containsPoint(center - offset) or \
                    rack.containsPoint(center + offset / 1.5) or rack.containsPoint(center - offset / 1.5):
                return other.group_id
        return self.nextGroupId()

    def scrollRequest(self, delta, orientation):
        units = - delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(value)
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        self.setZoom(self.zoomWidget.value() * increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def highlightSlice(self):
        """
        If the 3D viewer is active, highlight all the points that are in the current slice shown in annotation GUI.
        """
        if not self.pointcloud.viewer_is_ready():
            self.toggleActions(viewer=False)
            return
        self.pointcloud.highlight(self.pointcloud.select(indices=self.sliceIndices[self.sliceIdx],
                                                         showing=True, highlighted=False))

    def checkHighlightSlice(self):
        self.highlightSliceOnScroll = True

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.labelList.itemsToShapes:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def toggleIndivPolygon(self, item=None):
        if item is None:
            return
        shape = self.labelList.get_shape_from_item(item)
        if (self.canvas.getVisible(shape)):
            self.canvas.setShapeInvisible(shape)
            #item.setCheckState(Qt.Unchecked)
        else:
            self.canvas.setShapeVisible(shape)
            #item.setCheckState(Qt.Checked)
        #item.setCheckState(Qt.Checked if item.checkState() == Qt.Unchecked else Qt.Unchecked)

    def loadFile(self, filename):
        """
        Clear all information from last open file. Ask the user how many points to load and what mesh size and slice
        thickness. Load the point cloud data and build slice images from it. If there are existing annotations,
        load them.
        """
        labelfile = filename
        if filename.endswith('.json'):
            filename = LabelFile.get_source(filename)
        self.canvas.setEnabled(False)
        self.resetState()

        self.sourcePath = filename
        dialog = OpenFileDialog()
        if dialog.exec():
            self.max_points, self.scale, self.thickness = dialog.getInputs()
        else:
            return
        self.lastOpenDir = osp.dirname(filename)
        self.status(self.tr('Loading points from file'))
        self.loadPointCloud(filename)
        self.status(self.tr('Building pixel maps'))
        self.buildImageData()
        self.updatePixmap()
        self.loadLabelsFile(labelfile)
        self.setZoomAndScroll()
        self.canvas.setEnabled(True)

        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.status(self.tr("Loaded %s") % osp.basename(str(filename)))

    def setZoomAndScroll(self):
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config['keep_prev_scale']:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )

    def loadPointCloud(self, filename):
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr('Error opening file'),
                self.tr('No such file: <b>%s</b>') % filename
            )
            return False
        self.filename = str(filename)
        self.pointcloud.load(filename, self.max_points)
        self.status(self.tr("Loaded %s") % osp.basename(filename))

    def buildImageData(self, scores=None):
        # Divide the point cloud into horizontal slices with some thickness
        points = self.pointcloud.points[['x', 'y', 'z']].values
        min_point, max_point = points.min(axis=0), points.max(axis=0)
        min_idx, max_idx = (min_point / self.scale).astype(int), (max_point / self.scale).astype(int)
        slices = VoxelGrid(points, (10000., 10000., self.thickness))
        self.offset = QtCore.QPointF(min_point[0], min_point[1])
        # Create bitmaps (2D rectangular integer arrays) from the slices
        bitmaps = []
        self.sliceIndices = []
        self.status(self.tr('Building bitmaps from point cloud'))
        sliceList = slices.all()
        size = len(sliceList)
        for v in tqdm(slices.all(), desc='Building bitmaps from point cloud'):
            index = sliceList.index(v)
            percent = (index / size) * 100
            self.progressBar.setValue(percent)
            if not len(slices.indices(v)):
                continue
            self.sliceIndices.append(slices.indices(v))
            vg = VoxelGrid(self.pointcloud.points.loc[self.sliceIndices[-1]][['x', 'y', 'z']].values,
                           (self.scale, self.scale, max_point[2] + self.thickness / 2.0))
            if scores is not None:
                cur_scores = scores[self.sliceIndices[-1]]
            else:
                cur_scores = None
            bitmaps.append(vg.bitmapFromSlice(max=255, scores=cur_scores, min_idx=min_idx, max_idx=max_idx))
        # Create images from numpy arrays
        self.imageData = []
        self.progressBar.reset()
        for m in tqdm(bitmaps, desc='Building image data from bitmaps'):
            img = PIL.Image.fromarray(np.asarray(m, dtype="uint8"))
            buff = BytesIO()
            img.save(buff, format="JPEG")
            buff.seek(0)
            self.imageData.append(buff.read())

    def update3dViewer(self, values=None):
        if self.pointcloud.viewer_is_ready():
            self.pointcloud.render(showing=True)
            if values is not None:
                self.pointcloud.viewer.attributes(values)
        else:
            self.toggleActions(viewer=False)

    def updatePixmap(self, store=True):
        if not self.imageData:
            return
        if self.sliceIdx >= len(self.imageData):
            self.sliceIdx = 0
        if self.sliceIdx < 0:
            self.sliceIdx = len(self.imageData) - 1
        self.image = QtGui.QImage.fromData(self.imageData[self.sliceIdx])
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(self.image))
        self.canvas.loadShapes(self.labelList.shapes, store=store)
        if self.highlightSliceOnScroll:
            self.highlightSlice()

    def loadLabelsFile(self, filename):
        self.status(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + '.json'
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and \
                LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr('Error opening file'),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    ) % (e, label_file)
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.otherData = self.labelFile.otherData
            if 'roomName' not in self.otherData:
                pass
                self.roomNameDialog()
        else:
            self.labelFile = None

        if self._config['keep_prev']:
            prev_shapes = self.canvas.shapes
        if self._config['flags']:
            self.loadFlags({k: False for k in self._config['flags']})
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                self.loadFlags(self.labelFile.flags)
        if self._config['keep_prev'] and not self.labelList.shapes:
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config['store_data'] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue(
            'filename', self.filename if self.filename else '')
        self.settings.setValue('window/size', self.size())
        self.settings.setValue('window/position', self.pos())
        self.settings.setValue('window/state', self.saveState())
        self.settings.setValue('recentFiles', self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def pointsInShape(self, shape):
        """
        Return a boolean mask indicating which points from the pointcloud are contained in the given annotation.
        """
        if shape.label == 'beam' or shape.label == 'pole':
            keep = self.pointcloud.get_points_within(0.2, self.qpointToPointcloud(shape.points[0]), return_mask=True)
        elif shape.label == 'rectangle':
            box = [self.qpointToPointcloud(shape.points[0]), self.qpointToPointcloud(shape.points[1])]
            keep = self.pointcloud.in_box_2d(box)
        else:
            path = shape.makePath()
            keep = np.zeros(len(self.pointcloud.points), dtype=bool)
            for i, p in enumerate(self.pointcloud.points[['x', 'y']].values):
                keep[i] = path.contains(self.pointcloudToQpoint(p))
        return keep

    def nearestCrosshairIntersection(self, point, threshold=0.5):
        """
        Check to see if there is an intersection of beam crosshairs near the given point. If there are no crosshair
        lines nearby, return False along with the original point location. If there is a horizontal and/or vertical
        beam crosshair line nearby, return True with the coordinates of the nearby lines.
        """
        threshold = threshold / self.scale
        beams = []
        for item, shape in self.labelList.itemsToShapes:
            if shape.label == 'beam':
                beams.append((shape.points[0].x(), shape.points[0].y()))
        px, py = point.x(), point.y()
        intersection = QtCore.QPointF(px, py)
        intersected = False
        for x, y in beams:
            if abs(x - px) < threshold:
                intersection.setX(x)
                intersected = True
            if abs(y - py) < threshold:
                intersection.setY(y)
                intersected = True
        return intersection, intersected

    def showCrosshairs(self, value):
        for beam in self.beams:
            if value:
                beam.crosshairs = True
            else:
                beam.crosshairs = False
        self.updatePixmap()

    def interpolateBeamPositions(self):
        """
        Look at the existing beam positions and predict the location of all beam positions that we can predict based
        on the given information (within the canvas and walls). From these predictions, generate new annotations.
        """
        x, y = [], []
        beams = self.beams
        if not beams:
            return
        for beam in beams:
            x.append(beam.points[0].x())
            y.append(beam.points[0].y())
        self.remLabels(beams)
        x, y = np.unique(x, axis=0), np.unique(y, axis=0)
        dist_x = x[1] - x[0]
        dist_y = y[1] - y[0]
        # temp_x = x[0]
        # temp_y = y[0]
        # test_x, test_y = [], []
        # while temp_x < self.canvas.pixmap.width():
        #     temp_x += dist_x
        #     test_x.append(temp_x)
        # while temp_y < self.canvas.pixmap.height():
        #     temp_y += dist_y
        #     test_y.append(temp_y)
        # for cur_x in test_x:
        #     for cur_y in test_y:
        #         new_shape = Shape(label='beam', shape_type='point')
        #         new_shape.addPoint(QtCore.QPointF(cur_x, cur_y))
        #         new_shape.close()
        #         self.addLabel(new_shape)
        #         self.breakAllRacksWithBeam(new_shape)
        for cur_x in x:
            for cur_y in y:
                new_shape = Shape(label='beam', shape_type='point')
                new_shape.addPoint(QtCore.QPointF(cur_x, cur_y))
                new_shape.close()
                self.finalizeBeam(new_shape, snapToGrid=False, snapToPoints=False)
                self.addLabel(new_shape)

    def unbreakRack(self):
        """
        Merge the selected racks into a single rack that spans the space of all the racks combined.
        """
        racks = []
        for item in self.labelList.selectedItems():
            shape = self.labelList.get_shape_from_item(item)
            if 'rack' in shape.label:
                racks.append(shape)
        points = []
        for rack in racks:
            points.append(self.qpointToPointcloud(rack.points[0]))
            points.append(self.qpointToPointcloud(rack.points[1]))
        self.remLabels(racks[1:])
        racks[0].points[0] = self.pointcloudToQpoint(np.min(points, axis=0))
        racks[0].points[1] = self.pointcloudToQpoint(np.max(points, axis=0))
        self.finalizeRack(racks[0], tighten=True, orient=True, normalize=True)
        self.setDirty()
        self.updatePixmap()

    def breakRackManual(self, pos):
        """
        Break a rack using the mouse. [CTRL + right mouse button]
        """
        for item, shape in self.labelList.itemsToShapes:
            if 'rack' in shape.label and shape.containsPoint(pos):
                rack = shape
                if rack.orient % 2:
                    pos = pos.y()
                else:
                    pos = pos.x()
                break
        if rack is None:
            return
        else:
            self.breakRack(pos, rack)
            self.setDirty()
            self.updatePixmap()

    def breakAllRacksWithBeam(self, beam):
        """
        Given a beam, loop over all existing racks and break each one that is close enough to this beam.
        """
        for rack in self.racks:
            if self.canvas.isVisible(rack):
                breaks, pos, orient = self.beamBreaksRack(beam, rack)
                if breaks:
                    self.breakRack(pos, rack, orient)
        self.setDirty()
        self.updatePixmap()

    def breakRack(self, pos, rack, orientation=None, tighten=False, normalize=True):
        """
        Turn one rack into two racks at the given position. If an orientation is not chosen, then assume we are
        breaking a long rack into two shorter racks. Optionally tighten the resulting annotations around the points.
        Optionally resize the new racks to fit integer number of pallet locations.
        Delete any rack that is not large enough to fit a pallet.
        """
        new_rack = rack.copy()
        new_rack.rack_id = self.nextRackId()
        if orientation is None:
            orientation = rack.orient
        if orientation % 2 == 0:
            rack.points[1].setX(pos - 0.2)
            new_rack.points[0].setX(pos + 0.2)
        else:
            rack.points[1].setY(pos - 0.2)
            new_rack.points[0].setY(pos + 0.2)
        if self.isRackBigEnough(new_rack):
            self.addLabel(new_rack)
        self.finalizeRack(rack, tighten=tighten, normalize=normalize)
        self.finalizeRack(new_rack, tighten=tighten, normalize=normalize)
        return new_rack

    def finalizeRack(self, rack, tighten=False, orient=False, normalize=True):
        if tighten:
            self.tightenRack(rack)
        if orient:
            rack.orient = self.rackOrientation(rack)
        if normalize:
            self.normalizeRackDimensions(rack)
        rack.calculateRackExitEdge()
        if not self.isRackBigEnough(rack):
            self.remLabels([rack])

    def finalizeBeam(self, beam, snapToGrid=True, snapToPoints=True, breakRack=True):
        snapped = False
        if snapToGrid:
            intersection, snapped = self.nearestCrosshairIntersection(beam.points[0])
            if snapped:
                beam.points[0] = intersection
        if snapToPoints and not snapped:
            transformed = self.qpointToPointcloud(beam.points[0])
            snapped = self.pointcloud.snap_to_center(transformed, self._config['snap_center_thresh'])
            if not np.any(np.isnan(snapped)):
                beam.points[0] = self.pointcloudToQpoint(snapped)
        if breakRack:
            self.breakAllRacksWithBeam(beam)
        beam.lines = self.canvas.getEdges(beam)
        beam.crosshairs = self.actions.showCrosshairs.isChecked()
        if beam.label == 'I_beam':
            beam.point_type = Shape.P_SQUARE

    def breakAllRacks(self):
        size = len(self.beams)
        self.status(self.tr('Breaking all racks'))
        for beam in self.beams:
            index = self.beams.index(beam)
            percent = (index / size) * 100
            self.progressBar.setValue(percent)
            self.breakAllRacksWithBeam(beam)
        self.progressBar.reset()

    def isRackBigEnough(self, rack):
        bounds = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        dims = np.abs(bounds[1] - bounds[0])
        if rack.orient % 2:
            dims = np.flip(dims)
        return not (dims < np.array(self._config[rack.label][:2])).any()

    def tightenRack(self, rack):
        box = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        inbox = self.pointcloud.in_box_2d(box)
        if not np.sum(inbox):
            print('No points contained in this rack')
            return
        points = self.pointcloud.points.loc[inbox][['x', 'y', 'z']].values
        filtered = points[:, 2] > 3.0
        filtered[points[:, 2] > 7.0] = False
        if np.sum(filtered) > 1000:
            points = points[filtered]
        vg = VoxelGrid(points, (0.02, 0.02, 10000.0))
        scores = np.zeros(len(points))
        for v in vg.occupied():
            scores[vg.indices(v)] = vg.counts(v)
        filtered = scores > np.percentile(scores, 50)
        if np.sum(filtered):
            points = points[filtered]
            box = np.array((points.min(axis=0)[:2], points.max(axis=0)[:2]))
            rack.points[0], rack.points[1] = self.pointcloudToQpoint(box[0]), self.pointcloudToQpoint(box[1])
        else:
            self.remLabels([rack])

    def userTightenRack(self):
        item = self.labelList.selectedItems()[0]
        rack = self.labelList.get_shape_from_item(item)
        if 'rack' not in rack.label:
            return
        else:
            self.finalizeRack(rack, tighten=True)

    def predictPalletsForAllRacks(self):
        size = len(self.racks)
        self.status(self.tr('Predicting pallets for all racks'))
        for rack in self.racks:
            index = self.racks.index(rack)
            percent = (index / size) * 100
            self.progressBar.setValue(percent)
            self.predictPalletsFromRack(rack)
        self.setDirty()
        self.progressBar.reset()
        self.updatePixmap()

    def normalizeRackDimensions(self, rack):
        box = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        orient = rack.orient % 2
        unit_dims = np.flip(self._config[rack.label][:2]) if orient else np.array(self._config[rack.label][:2])
        total_dims = box[1] - box[0]
        discrete_dims = (total_dims / unit_dims).astype(int)
        discrete_dims += total_dims / unit_dims - discrete_dims > 0.6
        if rack.label == 'select_rack':
            discrete_dims[1-orient] = min(discrete_dims[1-orient], 1)
        box[1] = box[0] + unit_dims * discrete_dims
        rack.points[0], rack.points[1] = self.pointcloudToQpoint(box[0]), self.pointcloudToQpoint(box[1])

    def predictPalletsFromRack(self, rack):
        box = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        orient = rack.orient % 2
        dims = np.flip(self._config[rack.label][:2]) if orient else np.array(self._config[rack.label][:2])
        min_corner, max_corner = box[0] + np.array((0.0, 0.0)), box[0] + dims
        center = (min_corner + max_corner) / 2.0
        pallets = []
        while center[1 - orient] < box[1][1 - orient]:
            while center[orient] < box[1][orient]:
                pallet = rack.copy()
                pallet.label = 'pallet'
                pallet.points[0], pallet.points[1] = self.pointcloudToQpoint(min_corner), self.pointcloudToQpoint(max_corner)
                pallets.append(pallet)
                min_corner[orient], max_corner[orient], center[orient] = min_corner[orient] + dims[orient], max_corner[orient] + dims[orient], center[orient] + dims[orient]
            if 'select' in rack.label:
                break
            min_corner[orient], max_corner[orient] = box[0][orient], box[0][orient] + dims[orient]
            center[orient] = (min_corner[orient] + max_corner[orient]) / 2.0
            min_corner[1 - orient], max_corner[1 - orient], center[1 - orient] = min_corner[1 - orient] + dims[1 - orient], max_corner[1 - orient] + dims[1 - orient], center[1 - orient] + dims[1 - orient]
        for p in pallets:
            self.addLabel(p)

    def houghRackPosition(self, rack, threshold=0.5):
        # Filter points based on xy histogram
        box = [self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])]
        inbox = self.pointcloud.in_box_2d(box)
        points = self.pointcloud.points.loc[inbox][['x', 'y', 'z']].values
        vg = VoxelGrid(points, (0.02, 0.02, 10000.0))
        max_count = vg.counts(vg.fullest())
        keep = np.zeros(len(points), dtype=bool)
        for v in vg.occupied():
            if vg.counts(v) > max_count * 0.5:
                keep[vg.indices(v)] = True
        # Grab the dimensions of the rack with proper orientation
        dims = self._config[rack.label][:2]
        resolution = 0.05
        if rack.orient % 2:
            dims = np.flip(dims)
        from collections import defaultdict
        shift = (np.array(dims) / resolution / 2.0).astype(int)
        votes = defaultdict(int)
        for p in points:
            idx = (p / resolution).astype(int)
            for direction in [np.array((-1, -1)), np.array((-1, 1)), np.array((1, 1)), np.array((1, -1))]:
                votes[tuple(idx + direction * shift)] += 1
        max_votes = 0
        for key, value in votes.items():
            if value > max_votes:
                max_votes = value
        pallets = []
        for key, value in votes.items():
            if isinstance(key, tuple) and value > threshold * max_votes:
                pallets.append(np.array(key) * resolution)
        return pallets

    def isTwoRacks(self, rack):
        """
        Return True if the given rack annotation is actually large enough to be two back-to-back racks.
        """
        bounds = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        dims = np.abs(bounds[1] - bounds[0])
        return (dims / 1.9 > self._config[rack.label][1] * self._config[rack.label][2]).all()

    def breakBackToBackRacks(self, rack):
        """
        Break a single rack into two back-to-back racks.
        """
        bounds = np.array([self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])])
        dims = np.abs(bounds[1] - bounds[0])
        depth = self._config[rack.label][1] * self._config[rack.label][2]
        if abs(dims[0] - depth * 2.0) < abs(dims[1] - depth * 2.0):
            middle = (rack.points[0].x() + rack.points[1].x()) / 2.0
            new_rack = self.breakRack(middle, rack, 0, tighten=True)
        else:
            middle = (rack.points[0].y() + rack.points[1].y()) / 2.0
            new_rack = self.breakRack(middle, rack, 1, tighten=True)
        return new_rack

    def rotateShapes(self, angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))
        for s, shape in enumerate(self.canvas.shapes):
            for p, point in enumerate(shape.points):
                trans = self.qpointToPointcloud(point)
                trans = np.dot(trans, rot)
                self.canvas.shapes[s].points[p] = self.pointcloudToQpoint(trans)

    def rotateRack(self):
        items = self.labelList.selectedItems()
        if items:
            rack = self.labelList.get_shape_from_item(items[0])
            if 'rack' not in rack.label:
                return
            rack.orient += 1
            if rack.orient >= 4:
                rack.orient = 0
            self.setDirty()
            rack.calculateRackExitEdge()
            self.updatePixmap()

    def render3d(self):
        if not self.pointcloud.viewer_is_ready():
            self.pointcloud.render_flag = True
            self.pointcloud.viewer = None
        self.pointcloud.render()

    def qpointToPointcloud(self, p):
        return (p.x() * self.scale + self.offset.x(),
                (self.canvas.pixmap.height() - p.y()) * self.scale + self.offset.y())

    def pointcloudToQpoint(self, p):
        x = (p[0] - self.offset.x()) / self.scale
        y = self.canvas.pixmap.height() - ((p[1] - self.offset.y()) / self.scale)
        return QtCore.QPointF(x, y)

    def highlightWalls(self):
        walls = []
        for s in self.labelList.shapes:
            if s.label[:4] == 'wall':
                walls.append([(s.points[0].x(), s.points[0].y()), (s.points[1].x(), s.points[1].y())])

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def showNextSlice(self, _value=False):
        self.sliceIdx += 1
        self.updatePixmap(store=False)

    def showLastSlice(self, _value=False):
        self.sliceIdx -= 1
        self.updatePixmap(store=False)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else '.'
        formats = ['*.{}'.format(fmt.data().decode())
                   for fmt in QtGui.QImageReader.supportedImageFormats()]
        filters = self.tr("Image & Label files (%s)") % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr('%s - Choose Image or Label file') % __appname__,
            path, filters)
        if QT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.loadFile(filename)

    def openPointCloud(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else '.'
        formats = ['*.las', '*.laz', '*.pcd', '*.ply', '*.pts']
        filters = self.tr("Point Cloud files (%s)") % ' '.join(
            formats + ['*%s' % LabelFile.suffix])
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr('%s - Choose Point Cloud file') % __appname__, path, filters)
        if QT5:
            filename, _ = filename
        filename = str(filename)
        if filename:
            self.loadFile(filename)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('%s - Save/Load Annotations in Directory') % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr('%s . Annotations will be saved/loaded in %s') %
            ('Change Annotations Dir', self.output_dir))
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(
                self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self._config['flags'] or self.hasLabels():
            if self.labelFile:
                # DL20180323 - overwrite when in directory
                self._saveFile(self.labelFile.filename)
            elif self.output_file:
                self._saveFile(self.output_file)
                self.close()
            else:
                self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.hasLabels():
            self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr('%s - Choose File') % __appname__
        filters = self.tr('Label files (*%s)') % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self, self.tr('Choose File'), default_labelfile_name,
            self.tr('Label files (*%s)') % LabelFile.suffix)
        if QT5:
            filename, _ = filename
        filename = str(filename)
        return filename

    def roomNameDialog(self):
        dlg = QtWidgets.QInputDialog(self)
        dlg.setTextValue('room1')
        dlg.setLabelText('Room name')
        dlg.setWindowTitle('Set room name')
        if dlg.exec():
            self.otherData['roomName'] = dlg.textValue()

    def _saveFile(self, filename):
        if 'roomName' not in self.otherData:
            pass
            self.roomNameDialog()
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith('.json'):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + '.json'

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr('You are about to permanently delete this label file, '
                      'proceed anyway?')
        answer = mb.warning(self, self.tr('Attention'), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info('Label file is removed: {}'.format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if not self.labelList.itemsToShapes:
            self.errorMessage(
                'No objects labeled',
                'You must label at least one object to save the file.')
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename)
        answer = mb.question(self,
                             self.tr('Save annotations?'),
                             msg,
                             mb.Save | mb.Discard | mb.Cancel,
                             mb.Save)
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else '.'

    def toggleKeepPrevMode(self):
        self._config['keep_prev'] = not self._config['keep_prev']

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            'You are about to permanently delete {} polygons, '
            'proceed anyway?'
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
                self, self.tr('Attention'), msg,
                yes | no):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def highlightPointsInLabel(self, shape):
        if 'beam' in shape.label:
            point = np.array(self.qpointToPointcloud(shape.points[0]))
            box = [point - 0.1, point + 0.1]
        elif shape.label == 'pole':
            point = np.array(self.qpointToPointcloud(shape.points[0]))
            box = [point - 0.05, point + 0.05]
        elif 'rack' in shape.label or shape.label == 'pallet':
            box = [self.qpointToPointcloud(shape.points[0]), self.qpointToPointcloud(shape.points[1])]
        else:
            return
        inbox = self.pointcloud.in_box_2d(box)
        if not np.sum(inbox):
            return
        self.pointcloud.highlight(self.pointcloud.select(inbox, highlighted=False))

    def updateSelectedLabelWithHighlightedPoints(self):
        items = self.labelList.selectedItems()
        if len(items) != 1 or not self.pointcloud.viewer_is_ready():
            return
        shape = self.labelList.get_shape_from_item(items[0])
        points = self.pointcloud.points.loc[self.pointcloud.viewer.get('selected')][['x', 'y']].values
        if 'rack' in shape.label:
            shape.points[0] = self.pointcloudToQpoint(points.min(axis=0))
            shape.points[1] = self.pointcloudToQpoint(points.max(axis=0))
        elif 'beam' in shape.label or shape.label == 'pole':
            shape.points[0] = self.pointcloudToQpoint((points.min(axis=0) + points.max(axis=0)) / 2.0)

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) \
                if self.filename else '.'

        targetDirPath = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('%s - Open Directory') % __appname__,
            defaultOpenDirPath,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    @property
    def racks(self):
        racks = []
        for _, shape in self.labelList.itemsToShapes:
            if 'rack' in shape.label:
                racks.append(shape)
        return racks

    @property
    def beams(self):
        beams = []
        for _, shape in self.labelList.itemsToShapes:
            if 'beam' in shape.label:
                beams.append(shape)
        return beams

    @property
    def walls(self):
        for _, shape in self.labelList.itemsToShapes:
            if shape.label == 'walls':
                return shape

    @property
    def doors(self):
        doors = []
        for _, shape in self.labelList.itemsToShapes:
            if 'door' in shape.label:
                doors.append(shape)
        return doors

    @property
    def noise(self):
        noise = []
        for _, shape in self.labelList.itemsToShapes:
            if shape.label == 'noise':
                noise.append(shape)
        return noise

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = osp.splitext(filename)[0] + '.json'
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and \
                    LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QtGui.QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    def showRackHistogram(self, rack, axis='short'):
        """
        Display a histogram of the density of points of a given rack along the given axis. If axis is 'both', then
        display a 2D histogram of the rack.
        """
        import matplotlib.pyplot as plt
        box = [self.qpointToPointcloud(rack.points[0]), self.qpointToPointcloud(rack.points[1])]
        inbox = self.pointcloud.in_box_2d(box)
        if axis == 'both':
            x, y = self.pointcloud.points.loc[inbox][['x', 'y']].values.T
            plt.hist2d(x, y, bins=100)
        elif (axis == 'short' and rack.orient % 2) or (axis == 'long' and not rack.orient % 2):
            x = self.pointcloud.points.loc[inbox, 'x'].values
            plt.hist(x, bins=100)
        else:
            x = self.pointcloud.points.loc[inbox, 'y'].values
            plt.hist(x, bins=100)
        plt.show()

    def selectPalletsByRack(self):
        items = self.labelList.selectedItems()
        idx = None
        if items:
            for item in items:
                idx = self.labelList.get_shape_from_item(item).rack_id
                item.setSelected(False)
        if idx is None:
            return
        for item, shape in self.labelList.itemsToShapes:
            if shape.rack_id == idx and shape.label == 'pallet':
                item.setSelected(True)

    def selectPalletsByGroup(self):
        items = self.labelList.selectedItems()
        idx = None
        if items:
            for item in items:
                idx = self.labelList.get_shape_from_item(item).group_id
                item.setSelected(False)
        if idx is None:
            return
        for item, shape in self.labelList.itemsToShapes:
            if shape.group_id == idx and shape.label == 'pallet':
                item.setSelected(True)

    def selectBeams(self):
        for item, shape in self.labelList.itemsToShapes:
            if shape.label == 'beam':
                item.setSelected(True)
            else:
                item.setSelected(False)

    def selectBeamColumn(self):
        intersection, intersected = self.nearestCrosshairIntersection(self.canvas.prevPoint, 3.0)
        if not intersected:
            return
        for item, shape in self.labelList.itemsToShapes:
            if shape.label == 'beam' and shape.points[0].x() == intersection.x():
                item.setSelected(True)
            else:
                item.setSelected(False)

    def selectBeamRow(self):
        intersection, intersected = self.nearestCrosshairIntersection(self.canvas.prevPoint, 3.0)
        if not intersected:
            return
        for item, shape in self.labelList.itemsToShapes:
            if shape.label == 'beam' and shape.points[0].y() == intersection.y():
                item.setSelected(True)
            else:
                item.setSelected(False)

    def selectPallets(self):
        for item, shape in self.labelList.itemsToShapes:
            if shape.label == 'pallet':
                item.setSelected(True)
            else:
                item.setSelected(False)

    def resolveRackRectIntersection(self, rack, noise=False):
        """
        Find out where the given rack rectangle intersects with another rack (or noise) rectangle, figure out which
        dimension to crop, and crop it in order to resolve the intersection.
        """
        if noise:
            rectangles = self.noise
        else:
            rectangles = self.racks
        for other in rectangles:
            overlap_x, overlap_y = rack.rectIntersection(other)
            if not overlap_x * overlap_y:
                continue
            if overlap_x > overlap_y:
                if (other.points[0] + other.points[1]).y() < (rack.points[0] + rack.points[1]).y():
                    rack.points[1].setY(rack.points[1].y() + overlap_y)
                else:
                    rack.points[0].setY(rack.points[0].y() - overlap_y)
            else:
                if (other.points[0] + other.points[1]).x() < (rack.points[0] + rack.points[1]).x():
                    rack.points[0].setX(rack.points[0].x() + overlap_x)
                else:
                    rack.points[1].setX(rack.points[1].x() - overlap_x)

    def resolveRackWallIntersection(self, rack):
        """
        Find out where the given rack is intersecting a wall, calculate which direction the rack needs to be cropped
        in order to keep all the rack inside the walls, and perform the crop.
        """
        walls = self.walls
        if not walls:
            return

        def linesIntersect(p1, p2, p3, p4):
            r = p2 - p1
            s = p4 - p3
            d = r.x() * s.y() - r.y() * s.x()
            if abs(d) < 0.000001:
                return False
            u = ((p3.x() - p1.x()) * r.y() - (p3.y() - p1.y()) * r.x()) / d
            t = ((p3.x() - p1.x()) * s.y() - (p3.y() - p1.y()) * s.x()) / d
            if (0 <= u <= 1) and (0 <= t <= 1):
                return p1 + t * r
            else:
                return False

        rack_polygon = Shape.rectangleToPolygon(rack)

        for i in range(len(walls.points)):
            p1, p2 = walls.points[i], walls.points[(i+1) % len(walls.points)]
            for j in range(len(rack_polygon.points)):
                # j == 0, 1, 2, 3 --> left, top, right, bottom
                p3, p4 = rack_polygon.points[j], rack_polygon.points[(j+1) % len(rack_polygon.points)]
                intersection = linesIntersect(p1, p2, p3, p4)
                p3_inside = walls.containsPoint(p3)
                if intersection:
                    if (j == 0 and not p3_inside) or (j == 2 and p3_inside):
                        rack.points[0].setY(intersection.y())
                    elif (j == 0 and p3_inside) or (j == 2 and not p3_inside):
                        rack.points[1].setY(intersection.y())
                    elif (j == 1 and not p3_inside) or (j == 3 and p3_inside):
                        rack.points[0].setX(intersection.x())
                    elif (j == 1 and p3_inside) or (j == 3 and not p3_inside):
                        rack.points[1].setX(intersection.x())

    def annotate3dBeam(self):
        """
        Create a new beam annotation based on the currently highlighted points in the 3D viewer.
        """
        if self.pointcloud.viewer_is_ready():
            highlighted = self.pointcloud.get_highlighted_mask()
            if not highlighted.count():
                return
            points = self.pointcloud.points.loc[highlighted.bools][['x', 'y']].values
            min_p, max_p = points.min(axis=0), points.max(axis=0)
            center = (min_p + max_p) / 2.0
            beam = Shape('beam', shape_type='point')
            beam.addPoint(self.pointcloudToQpoint(center))
            beam.close()
            self.finalizeBeam(beam, snapToPoints=False)
            self.addLabel(beam)
            self.updatePixmap()

    def convertToFromIBeam(self):
        items = self.labelList.selectedItems()
        for item in items:
            shape = self.labelList.get_shape_from_item(item)
            if 'beam' in shape.label:
                if shape.label == 'beam':
                    shape.label = 'I_beam'
                    shape.point_type = Shape.P_SQUARE
                else:
                    shape.label = 'beam'
                    shape.point_type = Shape.P_ROUND
        if items:
            self.setDirty()

    def toggleRacks(self):
        self.renderingRacks = not self.renderingRacks
        if self.renderingRacks:
            for rack in self.racks:
                self.canvas.setShapeVisible(rack)
        else:
            for rack in self.racks:
                self.canvas.setShapeInvisible(rack)

    def togglePallets(self):
        self.renderingPallets = not self.renderingPallets
        if self.renderingPallets:
            for pallet in self.pallets:
                self.canvas.setShapeVisible(pallet)
        else:
            for pallet in self.pallets:
                self.canvas.setShapeInvisible(pallet)

    def toggleWalls(self):
        self.renderingWalls = not self.renderingWalls
        walls = self.walls
        if not walls:
            return
        if self.renderingWalls:
            self.canvas.setShapeVisible(walls)
        else:
            self.canvas.setShapeInvisible(walls)

    def toggleBeams(self):
        self.renderingBeams = not self.renderingBeams
        if self.renderingBeams:
            for beam in self.beams:
                self.canvas.setShapeVisible(beam)
        else:
            for beam in self.beams:
                self.canvas.setShapeInvisible(beam)
