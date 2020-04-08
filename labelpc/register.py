
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labelpc.pointcloud.PointCloud import PointCloud
from collections import defaultdict


class Room:

    def __init__(self, json_filename):
        self.json_file = json_filename
        self.annotations = []
        self.angle = 0.0
        self.offset = np.array((0.0, 0.0))
        with open(self.json_file) as f:
            self.json_data = json.load(f)
            self.source = self.json_data['sourcePath']
            self.name = self.json_data['roomName']
            self.annotations = pd.DataFrame(columns=['label', 'points'])
            for i, s in enumerate(self.json_data['shapes']):
                self.annotations.loc[i] = [s['label'], np.array(s['points'])]
        self.pointcloud = PointCloud(self.source, render=False, max_points=1000000)

    def rotate_annotations(self, angle, center=None):
        if center is None:
            center = np.zeros(2)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, s), (-s, c)))
        self.annotations['points'] -= center
        for idx, row in self.annotations.iterrows():
            row['points'] = np.dot(row['points'], rot)
        self.annotations['points'] += center

    def annotation_bounds(self):
        all_points = []
        for idx, row in self.annotations.iterrows():
            for p in row['points']:
                all_points.append(p)
        min_p, max_p = np.min(all_points, axis=0), np.max(all_points, axis=0)
        return min_p[0], min_p[1], max_p[0], max_p[1]

    def apply_registration(self, angle, offset):
        self.angle, self.offset = angle, offset
        self.rotate_annotations(angle)
        self.annotations['points'] += offset
        self.pointcloud.rotate_xy(angle)
        self.pointcloud.translate_xy(offset)

    def save(self):
        self.save_annotations()
        self.save_pointcloud()

    def save_annotations(self):
        with open(self.json_file.replace('.json', '_registered.json'), 'w') as f:
            for i in range(len(self.annotations)):
                self.json_data['shapes'][i]['points'] = self.annotations.loc[i, 'points'].tolist()
            basename, ext = self.source.split('.')
            self.json_data['sourcePath'] = basename + '_registered.' + ext
            json.dump(self.json_data, f)

    def save_pointcloud(self):
        basename, ext = self.source.split('.')
        outfile = basename + '_registered.' + ext
        full_cloud = PointCloud(self.source, max_points=1000000000, render=False)
        full_cloud.rotate_xy(self.angle)
        full_cloud.translate_xy(self.offset)
        full_cloud.reset_floor()
        full_cloud.write(outfile, overwrite=True)

    @property
    def doors(self):
        return np.unique(self.annotations.loc[self.annotations['label'].str.contains('door'), 'label'])


def intersection(room1, room2):
    # Get bounding boxes
    box_a = room1.annotation_bounds()
    box_b = room2.annotation_bounds()
    # Find number of points of each shape inside the other
    '''
    points_inside = 0
    for shape in room1.annotations['points']:
        for p in shape:
            if box_b[0] < p[0] < box_b[2] and box_b[1] < p[1] < box_b[3]:
                points_inside += 1
    for shape in room2.annotations['points']:
        for p in shape:
            if box_a[0] < p[0] < box_a[2] and box_a[1] < p[1] < box_a[3]:
                points_inside += 1
    '''
    points_inside = room1.pointcloud.in_box_2d(((box_b[0], box_b[1]), (box_b[2], box_b[3]))).sum()
    points_inside += room2.pointcloud.in_box_2d(((box_a[0], box_a[1]), (box_a[2], box_a[3]))).sum()
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    return max(0, xB - xA + 1) * max(0, yB - yA + 1) * (1.0 + points_inside)


def register_two_rooms(anchor_room, other_room):
    angle_res = 0.05
    best_angle, best_offset, min_cost = 0.0, np.array((0.0, 0.0)), np.inf
    if not np.any(anchor_room.annotations['label'] == 'door_%s' % other_room.name) or \
            not np.any(other_room.annotations['label'] == 'door_%s' % anchor_room.name):
        print('No matching doors for %s and %s' % (anchor_room.name, other_room.name))
        return best_angle, best_offset, min_cost
    for i in range(int(360.0 / angle_res)):
        anchor_doors = anchor_room.annotations.loc[anchor_room.annotations['label'] == 'door_%s' % other_room.name, 'points'].values
        anchor_orient = np.abs(anchor_doors[0][1] - anchor_doors[0][0])
        anchor_orient = anchor_orient[0] > anchor_orient[1]
        anchor_door_center = np.average([np.average(d, axis=0) for d in anchor_doors], axis=0)
        current_doors = other_room.annotations.loc[other_room.annotations['label'] == 'door_%s' % anchor_room.name, 'points'].values
        current_orient = np.abs(current_doors[0][1] - current_doors[0][0])
        current_orient = current_orient[0] > current_orient[1]
        current_door_center = np.average([np.average(d, axis=0) for d in current_doors], axis=0)
        if anchor_orient == current_orient:
            # Try moving the rooms toward and away from each other by 7 centimeters to account for wall thickness
            for offset_sign in [-1, 1]:
                offset = anchor_door_center - current_door_center  # Line doors up on top of each other
                offset[anchor_orient] += offset_sign * 0.07        # Shift doors away from each other by 7 cm
                other_room.annotations['points'] += offset
                cost = intersection(anchor_room, other_room)
                other_room.annotations['points'] -= offset
                if cost < min_cost:
                    min_cost = cost
                    best_offset = offset
                    best_angle = i * angle_res
        other_room.rotate_annotations(angle_res)
    return best_angle, best_offset, min_cost


def get_room_name_from_door(door):
    return door.split('_')[1]


def find_room_by_name(name, rooms):
    for room in rooms:
        if room.name == name:
            return room


def register(rooms):
    # Initially, all rooms are unanchored except for the anchor room
    anchored = defaultdict(bool)
    anchored[rooms[0].name] = True
    for room in rooms[1:]:
        anchored[room.name] = False
    # Loop over all the doors in the anchor room, anchor all connecting rooms first
    for door in rooms[0].doors:
        other_room_name = get_room_name_from_door(door)
        room = find_room_by_name(other_room_name, rooms)
        if room is not None:
            angle, offset, cost = register_two_rooms(rooms[0], room)
            print('anchoring', room.name, 'to', rooms[0].name, 'with cost:', cost)
            room.apply_registration(angle, offset)
            anchored[room.name] = True
    # Repeatedly loop over the list of rooms and anchor all rooms one by one to one another (max 10 iterations)
    count = 0
    while not np.all(anchored) and count < 10:
        # Find an anchored room
        for anchored_room in rooms:
            if not anchored[room.name]:
                continue
            # Loop over all the doors in this room to find a connecting room that is not anchored yet
            for door in room.doors:
                other_room_name = get_room_name_from_door(door)
                if anchored[other_room_name]:
                    continue
                other_room = find_room_by_name(other_room_name, rooms)
                if other_room is not None:
                    angle, offset, cost = register_two_rooms(anchored_room, other_room)
                    print('anchoring', other_room_name, 'to', anchored_room.name, 'with cost:', cost)
                    other_room.apply_registration(angle, offset)
                    anchored[other_room_name] = True
                    break
        count += 1


# Create scatter plot of points in a dataframe, but don't show the plot yet
def scatter_plot(room, color='blue'):
    x, y, c = [], [], []
    for idx, row in room.annotations.iterrows():
        for point in row['points']:
            x.append(point[0]), y.append(point[1])
            if 'door' in row['label']:
                c.append('red')
            else:
                c.append(color)
    plt.scatter(x, y, c=c)


# Create a scatter plot for each dataframe in dfs, and then show the final scatter plot
def show_points(rooms):
    colors = ['blue', 'green', 'cyan', 'yellow', 'orange', 'magenta', 'black', 'indigo']
    for i, room in enumerate(rooms):
        scatter_plot(room, colors[i])
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Please supply reference file and one or more register files')
        sys.exit()

    # Load the annotation data into a list of pandas dataframes
    rooms = []
    for fname in sys.argv[1:]:
        rooms.append(Room(fname))

    # Register the rooms (write data to new point cloud files)
    show_points(rooms)
    register(rooms)
    show_points(rooms)
    for room in rooms[1:]:
        room.save()

