from laspy.file import File
import numpy as np
import pandas as pd
import pptk
import os

from data.Mask import Mask
import algorithms.knn as knn
import algorithms.Features as feat
from data.Voxelize import VoxelGrid


class PointCloud:
    def __init__(self, filename, point_size=0.03, max_points=10000000, render=True):
        self.sample_las_file = '/home/brady/Data/RackBit.las'
        self.point_size = point_size
        self.max_points = max_points
        self.render_flag = render
        self.viewer = None
        self.las_header = None
        self.points = None
        self.showing = None
        self.index = None
        self.filename = filename
        if filename.endswith('.las'):
            self.__load_las_file(filename)
        elif filename.endswith('.ply'):
            self.__load_ply_file(filename)
        else:
            self.__load_xyz_file(filename)

    def __del__(self):
        if self.viewer:
            self.viewer.close()

    def __len__(self):
        return len(self.points)

    def __load_las_file(self, filename):
        with File(filename) as f:
            self.las_header = f.header.copy()
            if self.max_points is not None and self.max_points < f.header.point_records_count:
                mask = Mask(f.header.point_records_count, False)
                mask[np.random.choice(f.header.point_records_count, self.max_points)] = True
            else:
                mask = Mask(f.header.point_records_count, True)
            self.points = pd.DataFrame(np.array((f.x, f.y, f.z)).T[mask.bools])
            self.points.columns = ['x', 'y', 'z']
            if f.header.data_format_id >= 2:
                rgb = pd.DataFrame(np.array((f.red, f.green, f.blue), dtype='int').T[mask.bools])
                rgb.columns = ['r', 'g', 'b']
                self.points = self.points.join(rgb)
            self.points['class'] = f.classification[mask.bools]
            if np.sum(f.user_data):
                self.points['user_data'] = f.user_data[mask.bools].copy()
            if np.sum(f.intensity):
                self.points['intensity'] = f.intensity[mask.bools].copy()
        self.showing = Mask(len(self.points), True)
        self.render(self.showing)

    def __load_ply_file(self, filename, scale=1.0):
        import plyfile
        data = plyfile.PlyData.read(filename)
        self.points = pd.DataFrame()
        self.points['x'] = data.elements[0]['x']
        self.points['y'] = data.elements[0]['y']
        self.points['z'] = data.elements[0]['z']
        self.points['user_data'] = np.array(data.elements[0]['confidence'] * 255.0, dtype=int)
        self.points['intensity'] = np.array(data.elements[0]['intensity'] * 255.0, dtype=int)
        self.points['class'] = np.zeros(len(self.points), dtype=int)
        self.showing = Mask(len(self.points), True)
        self.render(self.showing)

        with File(self.sample_las_file) as f:
            self.las_header = f.header.copy()

    def __load_xyz_file(self, filename, indices=True):
        """
        This function allows the user to load point cloud data from an ascii text file format (extension xyz,
        txt, csv, etc.). The text file must have a header as the first line labeling the columns, and there must
        be at least 3 columns of data, the first one is unlabeled and gives the point indices, the second column
        is labeled 'x' and the third is labeled 'y'. Example header:  ',x,y,z,r,g,b,class'. If opening a file without
        a column of point indices, pass indices=False. If the text file was created using pandas, it will have the
        column of indices and the header; otherwise it will probably not have the indices column or the header.
        """
        if indices:
            self.points = pd.DataFrame.from_csv(filename)
        else:
            self.points = pd.read_csv(filename)
        if 'x' not in self.points or 'y' not in self.points:
            print('Error:  x and/or y missing from dataset. Please make sure there is x and y data in the point cloud',
                  'file, and that the file header indicates which columns store which attribute.')
            return
        if 'z' not in self.points:
            self.points['z'] = np.zeros(len(self.points))
        self.showing = Mask(len(self.points), True)
        self.render(self.showing)
        with File(self.sample_las_file) as f:
            self.las_header = f.header.copy()

    def render(self, mask=None, highlighted=False, showing=False):
        """
        This function allows the user to render some selection of the points to the viewer.
        By default, this function will render all points when called. If a mask is supplied, then those points
        will be rendered. If the highlighted or showing flags are True, then the appropriate selection will be used.
        :param mask: Mask object indicating which points to render.
        :param highlighted: If True, then render the currently highlighted points.
        :param showing: If True, then re-render all of the currently rendered points.
        """
        if not self.render_flag:
            return

        if mask is None:
            mask = self.select(highlighted=highlighted, showing=showing)

        if not mask.count():
            return

        self.showing.set(mask.bools)

        if not self.viewer:
            self.viewer = pptk.viewer(self.points.loc[mask.bools][['x', 'y', 'z']])
        else:
            self.viewer.clear()
            self.viewer.load(self.points.loc[mask.bools, ['x', 'y', 'z']])

        self.viewer.set(point_size=self.point_size, selected=[])
        if 'r' in self.points:
            scale = 255.0
            if 'user_data' in self.points and 'intensity' in self.points:
                self.viewer.attributes(self.points.loc[mask.bools, ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask.bools, 'class'],
                                       self.points.loc[mask.bools, 'user_data'],
                                       self.points.loc[mask.bools, 'intensity'])
            elif 'user_data' in self.points:
                self.viewer.attributes(self.points.loc[mask.bools, ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask.bools, 'class'],
                                       self.points.loc[mask.bools, 'user_data'])
            elif 'intensity' in self.points:
                self.viewer.attributes(self.points.loc[mask.bools, ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask.bools, 'class'],
                                       self.points.loc[mask.bools, 'intensity'])
            else:
                self.viewer.attributes(self.points.loc[mask.bools, ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask.bools, 'class'])
        else:
            self.viewer.attributes(self.points.loc[mask.bools, 'class'])

    def write(self, filename=None, mask=None, indices=None, highlighted=False, showing=False, overwrite=False, points=None):
        """
        This function allows the user to write out a subset of the current points to a LAS file.
        :param filename: Output filename to write to. Default is {current_filename}_out.las.
        :param mask: Mask object used to indicate which points to write to file.
        :param indices: List of integer indices indicating which points to write to file.
        :param highlighted: If True, then write the currently highlighted points to file.
        :param showing: If True, then write all the currently rendered points to file.
        :param overwrite: If True, then overwrite an existing file
        :param points: Pandas DataFrame containing the points and all the data to write. This DataFrame object must
        have x, y, and z attributes and optionally can have r, g, b, class, intensity, and user_data attributes.
        """
        if filename is None:
            filename = self.filename.split('.')[0] + '_out.las'
        elif not filename.endswith('.las'):
            filename += '.las'

        if os.path.exists(filename) and not overwrite:
            print(filename, 'already exists. Use option "overwrite=True" to overwrite')
            return

        if mask is None:
            mask = self.select(indices, highlighted, showing)

        with File(filename, self.las_header, mode='w') as f:
            if f.header.data_format_id < 2:
                f.header.data_format_id = 2

            if points is None:
                points = self.points.loc[mask.bools]

            f.x, f.y, f.z = points[['x', 'y', 'z']].values.T
            if 'r' in points:
                f.red, f.green, f.blue = points[['r', 'g', 'b']].values.T
            if 'class' in points:
                f.classification = points['class'].values
            if 'user_data' in points:
                f.user_data = points['user_data'].values
            if 'intensity' in points:
                f.intensity = points['intensity'].values

            print('Wrote %d points to %s' % (len(points), filename))

    def write_xyz(self, filename=None, mask=None, indices=None, highlighted=False, showing=False, points=None):
        """
        Write the selected points out to a csv file (text file format). All attributes will be written.
        """
        if mask is None:
            mask = Mask(len(self.points), True)
        if indices is not None:
            mask.setr(indices)
        if highlighted:
            mask.setr(self.viewer.get('selected'))
        if showing:
            mask.set(self.showing.bools)

        if filename is None:
            filename = self.filename.replace('.las', '.xyz')

        if points is None:
            self.points.loc[mask.bools].to_csv(filename)
        else:
            points.to_csv(filename)

    def write_ply(self, filename=None, mask=None, indices=None, highlighted=False, showing=False, points=None):
        """
        Write the selected points out to a csv file (text file format).
        """
        pass

    def get_relative_indices(self, mask, relative=None):
        """
        Return the chosen point indices relative to the currently rendered points (or some other set).
        """
        if relative is None:
            relative = self.showing
        mask.bools = mask.bools[relative.bools]
        return mask.resolve()

    def get_highlighted_mask(self):
        """
        Return a mask indicating which points are currently highlighted in the viewer.
        """
        mask = Mask(len(self.points), False)
        if self.viewer is not None:
            mask.setr_subset(self.viewer.get('selected'), self.showing)
        return mask

    def highlight(self, mask):
        """
        Set the selected points to be highlighted in the viewer (if they are rendered).
        """
        indices = self.get_relative_indices(mask)
        self.viewer.set(selected=indices)

    def get_perspective(self):
        """
        This function captures the current perspective of the viewer and returns its parameters so that the user
        can return to this perspective later or use it in a rendering sequence.
        :return: Perspective parameters (x, y, z, phi, theta, r).
        """
        x, y, z = self.viewer.get('eye')
        phi = self.viewer.get('phi')
        theta = self.viewer.get('theta')
        r = self.viewer.get('r')
        return [x, y, z, phi, theta, r]

    def set_perspective(self, p):
        """
        This method allows the user to set the camera perspective manually in the pptk viewer. It accepts a list as
        its single argument where the list defines the lookat position (x, y, z) the azimuthal angle, the elevation
        angle, and the distance from the lookat position. This list is returned from the method 'get_perspective()'.
        """
        self.viewer.set(lookat=p[0:3], phi=p[3], theta=p[4], r=p[5])

    def select(self, indices=None, highlighted=True, showing=False, classes=None, data=None, intensity=None,
               red=None, green=None, blue=None, compliment=False):
        """
        Return a mask indicating the selected points. Select points based on a number of methods including by
        index, by color, by class, by intensity, or by which points are rendered or highlighted in the viewer.
        If multiple selection methods are used at once, return the intersection of the selections.
        If compliment is True, then return everything EXCEPT the selected points.
        :param indices: List of point indices relative to the entire point cloud.
        :param highlighted: If True, then only grab the points currently highlighted in the viewer.
        :param showing: If True, then only grab points that are currently rendered.
        :param classes: Some list or iterable range from 0 to 31.
        :param data: Some list or iterable range from 0 to 255.
        :param intensity: Some list or iterable range from 0 to 255.
        :param red: Some list or iterable range from 0 to 255.
        :param green: Some list or iterable range from 0 to 255.
        :param blue: Some list or iterable range from 0 to 255.
        :param compliment: If True, return a mask indicating the NON-selected points.
        :return: Boolean mask indicating which points are selected relative to the full set.
        """

        if 'r' not in self.points:
            red, green, blue = None, None, None

        mask = Mask(len(self.points), True)
        cur_mask = Mask(len(self.points), False)
        if indices is not None and len(indices):
            mask.setr(indices)
        if highlighted:
            cur_mask = self.get_highlighted_mask()
            if cur_mask.count():
                mask.intersection(cur_mask.bools)
        if showing:
            mask.intersection(self.showing.bools)
        if classes is not None:
            cur_mask.false()
            for c in classes:
                cur_mask.union(self.points['class'] == c)
            mask.intersection(cur_mask.bools)
        if data is not None:
            cur_mask.false()
            for d in data:
                cur_mask.union(self.points['user_data'] == d)
            mask.intersection(cur_mask.bools)
        if intensity is not None:
            cur_mask.false()
            for i in intensity:
                cur_mask.union(self.points['intensity'] == i)
            mask.intersection(cur_mask.bools)
        if red is not None:
            cur_mask.false()
            for r in red:
                cur_mask.union(self.points['r'] == r)
            mask.intersection(cur_mask.bools)
        if green is not None:
            cur_mask.false()
            for g in green:
                cur_mask.union(self.points['g'] == g)
            mask.intersection(cur_mask.bools)
        if blue is not None:
            cur_mask.false()
            for b in blue:
                cur_mask.union(self.points['b'] == b)
            mask.intersection(cur_mask.bools)
        if compliment:
            mask.compliment()
        return mask

    def classify(self, cls, overwrite=False, mask=None):
        """
        Set the class of the currently selected points to cls. If the class is already set, then only
        overwrite the old value if "overwrite" is True.
        :param cls:
        :param overwrite:
        :param mask:
        :return:
        """
        if mask is None:
            mask = self.get_highlighted_mask()
        if overwrite:
            mask.intersection(self.points['class'] > 0)
        self.points.loc[mask.bools, 'class'] = cls
        self.render(showing=True)

    def neighbors(self, k=100, highlight=True):
        """
        Find the centroid of the currently highlighted points and return a Mask indicating which points are neighbors.
        :param k: Number of neighbors to find.
        :param highlight: If True, then set the currently highlighted points to the neareest k neighbors. Default True.
        :return: Return a copy of the mask indicating which points are neighbors of the selected point(s).
        """
        mask = self.select(showing=False, highlighted=True)
        if not mask.count() or mask.count() == len(self.points):
            print('No points were selected')
            return

        points = self.points.loc[mask.bools][['x', 'y', 'z']]
        if len(points) == 1:
            query = points
        else:
            query = np.average(points, axis=0)

        if self.index is None:
            self.index = knn.Query()
            self.index.pptk(self.points.loc[self.showing.bools][['x', 'y', 'z']].values)

        neighbors = self.index.neighbors(query, k)
        mask.setr(neighbors)
        if highlight:
            self.highlight(mask)
        return mask

    def rotate(self, points=None, degrees=0.0, axis=2):
        """
        This function takes in a list of points (or uses the currently rendered points) and returns a set of points that
        have been rotated by 'degrees' degrees about the given axis. By default, axis=2, so the points will rotate
        about the z-axis.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        else:
            points = points.copy()

        t = np.radians(degrees)
        rot = np.array(((np.cos(t), -np.sin(t)), (np.sin(t), np.cos(t))))
        if axis == 0:
            points[:, [1,2]] = np.dot(points[:, [1,2]], rot)
        elif axis == 1:
            points[:, [0,2]] = np.dot(points[:, [0,2]], rot)
        elif axis == 2:
            points[:, [0,1]] = np.dot(points[:, [0,1]], rot)

        return points

    def normals(self, points=None, k=100, r=0.35, render=False):
        """
        This function takes in a set of points (or uses the currently rendered points) and calculates the surface
        normals using pptk built-in functions which use PCA method. The number of neighbors (k) or the distance
        scale (r) can be changed to affect the resolution of the computation. If render=True, then the results
        will be rendered upon completion.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        n = np.abs(pptk.estimate_normals(points, k, r))
        if render:
            self.viewer.attributes(n)

        return n

    def curvature(self, points=None, k=100, r=0.35):
        """
        This function takes in a set of points (or uses the currently rendered points) and calculates the surface
        curvature using pptk built-in functions which use PCA method. The number of neighbors (k) or the distance
        scale (r) can be changed to affect the resolution of the computation. If render=True, then the results
        will be rendered upon completion.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        eigens = np.abs(pptk.estimate_normals(points, k, r, output_eigenvalues=True)[0])
        eigens.sort(axis=1)
        return eigens[:, 0] / eigens.sum(axis=1) * 3.0

    def curvature_voxelized(self, points=None, mesh=0.05):
        """
        This function takes in a set of points (or uses the currently rendered points) and calculates the surface
        curvature using PCA method and using voxelization as the neighbor calculation method. The mesh size parameter
        (mesh) allows the user to adjust the spatial resolution of the computation. If render=True, then the results
        will be rendered upon completion.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        vg = VoxelGrid(points, mesh_size=mesh)
        curvs = np.zeros(len(points))
        for indices in vg.indices():
            curvs[indices] = feat.curvature_from_eigenvalues(feat.eigenvalues_single(points[indices]))

        return curvs
