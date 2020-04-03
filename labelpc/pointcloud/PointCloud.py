from laspy.file import File
import open3d as o3d
import numpy as np
import pandas as pd
import pptk
import os

from labelpc.pointcloud.Mask import Mask
from labelpc.pointcloud.Voxelize import VoxelGrid


class PointCloud:
    def __init__(self, filename=None, point_size=0.01, max_points=10000000, render=True):
        self.point_size = point_size
        self.max_points = max_points
        self.render_flag = render
        self.viewer = None
        self.las_header = None
        self.points = pd.DataFrame(columns=['x', 'y', 'z', 'class'])
        self.showing = None
        self.index = None
        self.filename = filename
        if filename is None:
            self.render_flag = False
        else:
            self.load(filename)

    def __del__(self):
        if self.viewer:
            self.viewer.close()

    def __len__(self):
        return len(self.points)

    def load(self, filename, max_points=None):
        if max_points is not None:
            self.max_points = max_points
        if filename.endswith('.las') or filename.endswith('.laz'):
            self.__load_las_file(filename)
        elif filename.endswith('.ply'):
            self.__load_ply_file(filename)
        elif filename.endswith('.pcd'):
            self.__load_pcd_file(filename)
        elif filename.endswith('.xyz') or filename.endswith('.pts') or filename.endswith('.txt'):
            self.__load_xyz_file(filename)
        else:
            print('Cannot load %s: file type not supported' % filename)
            return
        self.showing = Mask(len(self.points), True)
        self.render(self.showing)

    def __from_open3d_point_cloud(self, cloud):
        new_df = pd.DataFrame(np.asarray(cloud.points), columns=['x', 'y', 'z'])
        if cloud.has_normals():
            normals = np.asarray(cloud.normals)
            new_df['r'] = (normals[:, 0] * 255.).astype(int)
            new_df['g'] = (normals[:, 1] * 255.).astype(int)
            new_df['b'] = (normals[:, 2] * 255.).astype(int)
        if cloud.has_colors():
            colors = np.asarray(cloud.colors)
            if colors[:, 0].max() > 0:
                new_df['class'] = (colors[:, 0] * 31.).astype(int)
            if colors[:, 1].max() > 0:
                new_df['user_data'] = (colors[:, 1] * 255.).astype(int)
            if colors[:, 2].max() > 0:
                new_df['intensity'] = (colors[:, 2] * 255.).astype(int)
        if self.max_points is not None and self.max_points < len(new_df):
            new_df = new_df.loc[np.random.choice(len(new_df), self.max_points)]
        return new_df

    def __unzip_laz(self, infile, outfile=None):
        import subprocess
        if outfile is None:
            outfile = infile.replace('.laz', '.las')
        args = ['laszip', '-i', infile, '-o', outfile]
        subprocess.run(" ".join(args), shell=True, stdout=subprocess.PIPE)

    def __zip_las(self, infile, outfile=None):
        import subprocess
        if outfile is None:
            outfile = infile.replace('.las', '.laz')
        args = ['laszip', '-i', infile, '-o', outfile]
        subprocess.run(" ".join(args), shell=True, stdout=subprocess.PIPE)

    def __load_las_file(self, filename):
        if filename.endswith('.laz'):
            orig_filename = filename
            filename = 'TEMPORARY.las'
            self.__unzip_laz(orig_filename, filename)
        with File(filename) as f:
            if self.las_header is None:
                self.las_header = f.header.copy()
            if self.max_points is not None and self.max_points < f.header.point_records_count:
                mask = Mask(f.header.point_records_count, False)
                mask[np.random.choice(f.header.point_records_count, self.max_points)] = True
            else:
                mask = Mask(f.header.point_records_count, True)
            new_df = pd.DataFrame(np.array((f.x, f.y, f.z)).T[mask.bools])
            new_df.columns = ['x', 'y', 'z']
            if f.header.data_format_id >= 2:
                rgb = pd.DataFrame(np.array((f.red, f.green, f.blue), dtype='int').T[mask.bools])
                rgb.columns = ['r', 'g', 'b']
                new_df = new_df.join(rgb)
            new_df['class'] = f.classification[mask.bools]
            if np.sum(f.user_data):
                new_df['user_data'] = f.user_data[mask.bools].copy()
            if np.sum(f.intensity):
                new_df['intensity'] = f.intensity[mask.bools].copy()
        self.points = self.points.append(new_df, sort=False)
        if filename == 'TEMPORARY.las':
            os.system('rm TEMPORARY.las')

    def __load_ply_file(self, filename):
        points = o3d.io.read_point_cloud(filename)
        self.points = self.points.append(self.__from_open3d_point_cloud(points), sort=False)
        """
        import plyfile
        data = plyfile.PlyData.read(filename)
        new_df = pd.DataFrame()
        new_df['x'] = data.elements[0]['x']
        new_df['y'] = data.elements[0]['y']
        new_df['z'] = data.elements[0]['z']
        new_df['user_data'] = np.array(data.elements[0]['confidence'] * 255.0, dtype=int)
        new_df['intensity'] = np.array(data.elements[0]['intensity'] * 255.0, dtype=int)
        new_df['class'] = np.zeros(len(new_df), dtype=int)
        self.points = self.points.append(new_df, sort=False)
        """

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
            new_df = pd.DataFrame.from_csv(filename)
        else:
            new_df = pd.read_csv(filename)
        if 'x' not in new_df or 'y' not in new_df:
            print('Error:  x and/or y missing from dataset. Please make sure there is x and y data in the point cloud',
                  'file, and that the file header indicates which columns store which attribute.')
            return
        if 'z' not in new_df:
            self.points['z'] = np.zeros(len(self.points))
        new_df['class'] = np.zeros(len(new_df), dtype=int)
        self.points = self.points.append(new_df, sort=False)

    def __load_pcd_file(self, filename):
        points = o3d.io.read_point_cloud(filename)
        self.points = self.points.append(self.__from_open3d_point_cloud(points), sort=False)
        """
        # Fixme: pypcd only works if you install using --> pip3 install --upgrade git+https://github.com/klintan/pypcd.git
        from pypcd import pypcd
        pcd = pypcd.PointCloud.from_path(filename)
        new_df = pd.DataFrame()
        new_df['x'] = pcd.pc_data['x']
        new_df['y'] = pcd.pc_data['y']
        new_df['z'] = pcd.pc_data['z']
        new_df['class'] = np.zeros(len(new_df), dtype=int)
        new_df = new_df.fillna(0)
        print(new_df)
        self.points = self.points.append(new_df, sort=False)
        """

    def __to_open3d_point_cloud(self, df):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
        if 'r' in df.columns:
            cloud.normals = o3d.utility.Vector3dVector(df[['r', 'g', 'b']].values / 255.)
        colors = np.zeros((len(df), 3))
        colors[:, 0] = df['class'] / 31.
        if 'user_data' in df.columns:
            colors[:, 1] = df['user_data'] / 255.
        if 'intensity' in df.columns:
            colors[:, 2] = df['intensity'] / 255.
        if colors.max() > 0:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def write(self, filename=None, mask=None, indices=None, highlighted=False, showing=False, overwrite=False, points=None):
        """
        This function allows the user to write out a subset of the current points to a file.
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
            filename = self.filename
        if os.path.exists(filename) and not overwrite:
            print(filename, 'already exists. Use option "overwrite=True" to overwrite')
            return
        if mask is None and points is None:
            mask = self.select(indices, highlighted, showing)
        if points is None:
            points = self.points.loc[mask.bools]
        if filename.endswith('.las') or filename.endswith('.laz'):
            self.__write_las_file(filename, points)
        elif filename.endswith('.ply'):
            self.__write_ply_file(filename, points)
        elif filename.endswith('.pcd'):
            self.__write_pcd_file(filename, points)
        elif filename.endswith('.xyz') or filename.endswith('.pts') or filename.endswith('.txt'):
            self.__write_xyz_file(filename, points)
        else:
            print('Unrecognized file type. Please use .las, .ply, .pcd, .xyz, .pts, or .txt.')
            return
        print('Wrote %d points to %s' % (len(points), filename))

    def __write_las_file(self, filename, points):
        if filename.endswith('.laz'):
            orig_filename = filename
            filename = 'TEMPORARY.las'
        if self.las_header.data_format_id < 2:
            self.las_header.data_format_id = 2
        with File(filename, self.las_header, mode='w') as f:
            f.x, f.y, f.z = points[['x', 'y', 'z']].values.T
            if 'r' in points:
                f.red, f.green, f.blue = points[['r', 'g', 'b']].values.T
            if 'class' in points:
                f.classification = points['class'].values
            if 'user_data' in points:
                f.user_data = points['user_data'].values
            if 'intensity' in points:
                f.intensity = points['intensity'].values
        if filename == 'TEMPORARY.las':
            self.__zip_las('TEMPORARY.las', orig_filename)
            os.system('rm TEMPORARY.las')

    def __write_xyz_file(self, filename, points):
        points.to_csv(filename)

    def __write_ply_file(self, filename, points):
        cloud = self.__to_open3d_point_cloud(points)
        o3d.io.write_point_cloud(filename, cloud)

    def __write_pcd_file(self, filename, points):
        cloud = self.__to_open3d_point_cloud(points)
        o3d.io.write_point_cloud(filename, cloud)
        """
        from pypcd import pypcd
        pc = pypcd.make_xyz_point_cloud(points.values)
        pc.save_pcd(filename)
        """

    def prepare_viewer(self, render_flag=None):
        """
        Check to see if the viewer is ready to receive commands. If it isn't, get it ready and return True.
        If the render flag is False, return False.
        """
        if render_flag is not None:
            self.render_flag = render_flag
        if not self.render_flag:
            return False
        if not self.viewer_is_ready():
            self.render(showing=True)
            return True

    def viewer_is_ready(self):
        """
        Return True if the viewer is ready to receive commands, else return False.
        """
        if not self.render_flag or not self.viewer:
            return False
        try:
            self.viewer.get('lookat')
            return True
        except ConnectionRefusedError:
            return False

    def close_viewer(self):
        if self.viewer_is_ready():
            self.viewer.close()
        self.viewer = None

    def render(self, mask=None, indices=None, highlighted=False, showing=False):
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
            mask = self.select(indices=indices, highlighted=highlighted, showing=showing)

        if not mask.count():
            return

        self.showing.set(mask.bools)

        if self.viewer_is_ready():
            self.viewer.clear()
            self.viewer.load(self.points.loc[mask.bools, ['x', 'y', 'z']])
        else:
            self.viewer = pptk.viewer(self.points.loc[mask.bools][['x', 'y', 'z']])

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
        if self.viewer_is_ready():
            mask.setr_subset(self.viewer.get('selected'), self.showing)
        return mask

    def highlight(self, mask=None, indices=None):
        """
        Set the selected points to be highlighted in the viewer (if they are rendered).
        """
        if mask is None and indices is None:
            return
        if indices is not None:
            mask = self.select(indices=indices, highlighted=False)
        if self.viewer_is_ready():
            indices = self.get_relative_indices(mask)
            self.viewer.set(selected=indices)

    def get_perspective(self):
        """
        This function captures the current perspective of the viewer and returns its parameters so that the user
        can return to this perspective later or use it in a rendering sequence.
        :return: Perspective parameters (x, y, z, phi, theta, r).
        """
        if not self.viewer_is_ready():
            return [0, 0, 0, 0, 0, 0]
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
        if self.viewer_is_ready():
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
        """
        if mask is None:
            mask = self.get_highlighted_mask()
        if overwrite:
            mask.intersection(self.points['class'] > 0)
        self.points.loc[mask.bools, 'class'] = cls
        self.render(showing=True)

    def center(self):
        """
        Shift the origin of the point cloud to its centroid.
        """
        self.points[['x', 'y', 'z']] -= np.average(self.points[['x', 'y', 'z']], axis=0)

    def reset_origin(self):
        """
        Shift the origin of the point cloud to the minimum of the point cloud.
        """
        self.points[['x', 'y', 'z']] -= self.points[['x', 'y', 'z']].values.min(axis=0)

    def rotate_xy(self, angle):
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array(((c, s), (-s, c)))
        self.points[['x', 'y']] = np.dot(self.points[['x', 'y']].values, rot)

    def translate_xy(self, offset):
        self.points[['x', 'y']] += offset

    def subsample(self, n=10000000, percent=1.0):
        """
        Return a random sample of the point cloud.
        """
        threshold = int(percent * len(self.points))
        if n < threshold:
            threshold = n
        if threshold < len(self.points):
            keep = np.zeros(len(self.points), dtype=bool)
            keep[np.random.choice(len(self.points), threshold)] = True
            return keep
        else:
            return np.ones(len(self.points), dtype=bool)

    def add_points(self, points):
        """
        Append a pandas dataframe of points to the current list of points. The pandas DataFrame
        must have 'x' and 'y' columns, and cannot have any abnormal columns in it.
        """
        if not isinstance(points, pd.DataFrame):
            print('Error: points must be in the form of a pandas DataFrame. Cannot append.')
            return
        if 'x' not in points.columns or 'y' not in points.columns:
            print('Error: missing x and/or y column data. Cannot append.')
            return
        for c in points.columns:
            if c not in self.points.columns:
                print('Error: unknown column', c, 'in points. Cannot append.')
                return
        self.points = self.points.append(points)
        self.points = self.points.fillna(0)

    def slice(self, points=None, position=1.75, thickness=0.2, axis=2):
        """
        Take a planar slice of some thickness out of the data. The slice will be axis-aligned.
        :param points: Set of points to take slice from. Default is all currently rendered points.
        :param position: Position along axis to take slice from. Default is 1.75, set for slicing vertical poles above pallets.
        :param thickness: Thickness of slice to take. Default is 0.2 (20 cm).
        :param axis: Axis perpendicular to the slice of data. Must be 0, 1, or 2 (default is 2 (z-axis)).
        """
        if axis == 2:
            str_axis = 'z'
        elif axis == 1:
            str_axis = 'y'
        else:
            str_axis = 'x'
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        mask = points[str_axis] > position
        mask[points[str_axis] > position + thickness] = False
        return mask

    def in_box_2d(self, box, points=None):
        if points is None:
            points = self.points[['x', 'y']].values
        keep = points > np.array(np.min(box, axis=0))
        keep[points > np.array(np.max(box, axis=0))] = False
        return keep.all(axis=1)

    @staticmethod
    def make_box_from_point(point, delta):
        point = np.array(point)
        return [point - delta, point + delta]

    def get_points_within(self, delta, point=None, return_mask=False, return_z=False):
        """
        Returns all the points within delta of the given point. Z-axis is not considered in distance calculation.
        If point is None, then use the currently highlighted point as the query point. If multiple points are currently
        highlighted, then use their average as the query point.
        """
        if point is None:
            selected = self.get_highlighted_mask()
            point = np.average(self.points.loc[selected.bools][['x', 'y']], axis=0)
        x, y = point[:2]
        keep = np.ones(len(self.points), dtype=bool)
        keep[self.points['x'] < x - delta] = False
        keep[self.points['x'] > x + delta] = False
        keep[self.points['y'] < y - delta] = False
        keep[self.points['y'] > y + delta] = False
        if return_mask:
            return keep
        elif return_z:
            return self.points.loc[keep][['x', 'y', 'z']].values
        else:
            return self.points.loc[keep][['x', 'y']].values

    def distance_to_line(self, line, point):
        p1, p2 = line
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        num = np.abs(dy * point[0] - dx * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
        den = np.sqrt(np.square(dx) + np.square(dy))
        return num / den

    def get_points_under_line(self, line, delta=0.01):
        keep = np.zeros(len(self.points), dtype=bool)
        for i, p in self.points:
            if self.distance_to_line(line, p) < delta:
                keep[i] = True
        return np.arange(len(self.points))[keep]

    def snap_to_corner(self, point, delta):
        return self.snap_to_center(point, delta)

    def snap_to_center(self, point, delta):
        points = self.get_points_within(delta, point, return_z=True)
        points = points[points[:, 2] > 0.5]
        points = points[points[:, 2] < 7.0]
        if len(points) > 100:
            return np.average(points, axis=0)[:2]
        else:
            return point

    def tighten_to_rack(self, box):
        points = self.points.loc[self.in_box_2d(box, self.points[['x', 'y']].values)][['x', 'y', 'z']].values
        filtered = points[:, 2] > 3.0
        filtered[points[:, 2] > 7.0] = False
        if np.sum(filtered) > 1000:
            points = points[filtered]
        vg = VoxelGrid(points, (0.02, 0.02, 10000.0))
        scores = np.zeros(len(points))
        for v in vg.occupied():
            scores[vg.indices(v)] = vg.counts(v)
        filtered = scores > np.percentile(scores, 50)
        points = points[filtered]
        return np.array((points.min(axis=0)[:2], points.max(axis=0)[:2]))

    def reset_floor(self, align_z=False):
        # Grab the floor points
        subset = np.random.choice(len(self.points), min(100000, len(self.points)))
        points = self.points.loc[subset][['x', 'y', 'z']].values
        vg = VoxelGrid(points, (10000.0, 10000.0, 0.02))
        floor_points = points[vg.indices(vg.fullest())]
        floor_centroid = floor_points.mean(axis=0)
        self.points['z'] -= floor_centroid[2]
        if align_z:
            dx, dy, dz = floor_points.max(axis=0) - floor_points.min(axis=0)
            if dx or dy:
                centroid = points.mean(axis=0)
                self.points[['x', 'y', 'z']] -= centroid
                if dx:
                    theta = -np.arctan(dz / dx)
                    c, s = np.cos(theta), np.sin(theta)
                    rot = np.array(((c, s), (-s, c)))
                    self.points[['x', 'z']] = np.dot(self.points[['x', 'z']].values, rot)
                if dy:
                    theta = -np.arctan(dz / dy)
                    c, s = np.cos(theta), np.sin(theta)
                    rot = np.array(((c, s), (-s, c)))
                    self.points[['y', 'z']] = np.dot(self.points[['y', 'z']].values, rot)
                self.points[['x', 'y', 'z']] += centroid
