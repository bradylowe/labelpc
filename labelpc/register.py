
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labelpc.pointcloud.PointCloud import PointCloud


def find_in_dataframe(df, query='door', indices=False):
    if indices:
        return df.index[df['label'].str.contains(query)].tolist()
    else:
        return df['label'].str.contains(query)


# Doors must have common name theme. (DOOR before DOCK, DOOR1 before DOOR2)
def get_common_door(ref, reg):
    ref_doors = ref.loc[ref['label'].str.contains('door')]
    for label in ref_doors['label'].unique():
        doors = reg.loc[reg['label'].str.contains(label)]
        if len(doors):
            return label


def distance(p1, p2):
    return np.sqrt(np.dot(p1 - p2, p1 - p2))


# Return the average distance from each door in one scan to the nearest door in the other scan
def door_loss(door, ref, reg):
    total, count = 0.0, 0
    for ref_points in ref.loc[ref['label'] == door, 'points']:
        min_dist = 1000000.0
        for reg_points in reg.loc[reg['label'] == door, 'points']:
            ref_d, reg_d = np.abs(ref_points[1] - ref_points[0]), np.abs(reg_points[1] - reg_points[0])
            if (ref_d[0] > ref_d[1]) == (reg_d[0] > reg_d[1]):
                ref_c, reg_c = ref_points.mean(axis=0), reg_points.mean(axis=0)
                min_dist = min(min_dist, distance(ref_c, reg_c))
        total += min_dist
        count += 1
    return total / count


def rotate_dataframe(df, angle, center=None):
    if center is None:
        center = np.zeros(2)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array(((c, s), (-s, c)))
    df['points'] -= center
    for idx, row in df.iterrows():
        row['points'] = np.dot(row['points'], rot)
    df['points'] += center
    return df


def dataframe_bounds(df):
    all_points = []
    for idx, row in df.iterrows():
        for p in row['points']:
            all_points.append(p)
    min_p, max_p = np.min(all_points, axis=0), np.max(all_points, axis=0)
    return min_p[0], min_p[1], max_p[0], max_p[1]


def intersection(df1, df2):
    boxA = dataframe_bounds(df1)
    boxB = dataframe_bounds(df2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    return max(0, xB - xA + 1) * max(0, yB - yA + 1)


# Measure how well two dataframes are registered together, 0.0 is perfect score
def cost_two_dfs(df1, df2, door=None):
    if door is None:
        door = get_common_door(df1, df2)
    return door_loss(door, df1, df2) * (1.0 + intersection(df1, df2))


# Measure how well many dataframes are registered together, 0.0 is perfect score
def cost_many_dfs(dfs):
    cost = 0.0
    for df1 in dfs:
        for df2 in dfs:
            if df1 is df2:
                continue
            cost += cost_two_dfs(df1, df2)
    return cost


def register_two_dfs(ref, reg, ref_name, reg_name):
    angle_res = 90.0
    best_angle, best_offset, min_cost = None, None, 1000000.0
    if not np.any(ref['label'] == 'door_%s' % reg_name) or not np.any(reg['label'] == 'door_%s' % ref_name):
        return best_angle, best_offset, min_cost
    for i in range(int(360.0 / angle_res)):
        anchor_doors = ref.loc[ref['label'] == 'door_%s' % reg_name, 'points'].values
        anchor_orient = np.abs(anchor_doors[0][1] - anchor_doors[0][0])
        anchor_orient = anchor_orient[0] > anchor_orient[1]
        anchor_door_center = np.average([np.average(d, axis=0) for d in anchor_doors], axis=0)
        current_doors = reg.loc[reg['label'] == 'door_%s' % ref_name, 'points'].values
        current_orient = np.abs(current_doors[0][1] - current_doors[0][0])
        current_orient = current_orient[0] > current_orient[1]
        current_door_center = np.average([np.average(d, axis=0) for d in current_doors], axis=0)
        if anchor_orient == current_orient:
            offset = anchor_door_center - current_door_center
            reg['points'] += offset
            cost = intersection(ref, reg)
            reg['points'] -= offset
            if cost < min_cost:
                min_cost = cost
                best_offset = offset
                best_angle = i * angle_res
        rotate_dataframe(reg, angle_res)
    return best_angle, best_offset, min_cost


def get_names_from_dataframes(dfs):
    names = []
    for df in dfs:
        doors = df.loc[df['label'].str.contains('door')]
        possible_names = doors[0].split('_')[1:]
        for name in possible_names:
            found_name = True
            for door in doors:
                if name not in door.split('_')[1:]:
                    found_name = False
                    break
            if found_name:
                names.append(name)
                break
    return names


def get_doors_from_dataframes(dfs):
    doors = []
    for df in dfs:
        doors.append(np.unique(df.loc[df['label'].str.contains('door'), 'label']))
    return doors


def get_room_name_from_door(door):
    return door.split('_')[1]


def register(dfs, names, sources):
    # Initially, all rooms are unanchored except for the anchor room
    anchored = np.zeros(len(dfs), dtype=bool)
    anchored[0] = True
    doors = get_doors_from_dataframes(dfs)
    # Loop over all the doors in the anchor room, anchor all connecting rooms first
    for door in doors[0]:
        other_room = get_room_name_from_door(door)
        if other_room not in names:
            continue
        other_idx = names.index(other_room)
        angle, offset, cost = register_two_dfs(dfs[0], dfs[other_idx], names[0], other_room)
        print('anchoring', names[other_idx], 'to', names[0], 'with cost:', cost)
        apply_registration_to_dataframe(dfs[other_idx], angle, offset)
        #apply_registration_to_pointcloud(sources[other_idx], angle, offset)
        anchored[other_idx] = True
    # Repeatedly loop over the list of rooms and anchor each room to an anchored room
    count = 0
    while not anchored.all() and count < 10:
        for i in range(len(anchored)):
            if anchored[i]:
                continue
            # Loop over all the doors in this room
            for door in doors[i]:
                other_room = get_room_name_from_door(door)
                anchor_idx = names.index(other_room)
                # If the room that this room connects to via current door is NOT anchored, skip this door
                if not anchored[anchor_idx]:
                    continue
                angle, offset, cost = register_two_dfs(dfs[anchor_idx], dfs[i], other_room, names[i])
                # If the cost is low enough, apply the alignment and consider the current door anchored
                if cost < 1000.0:
                    print('anchoring', names[i], 'to', names[anchor_idx], 'with cost:', cost)
                    apply_registration_to_dataframe(dfs[i], angle, offset)
                    #apply_registration_to_pointcloud(sources[i], angle, offset)
                    anchored[i] = True
                    break
        count += 1


# Create scatter plot of points in a dataframe, but don't show the plot yet
def scatter_plot(df, color='blue'):
    x, y, c = [], [], []
    for idx, row in df.iterrows():
        for point in row['points']:
            x.append(point[0]), y.append(point[1])
            if 'door' in row['label']:
                c.append('red')
            else:
                c.append(color)
    plt.scatter(x, y, c=c)


# Create a scatter plot for each dataframe in dfs, and then show the final scatter plot
def show_points(dfs):
    colors = ['blue', 'green', 'cyan', 'yellow', 'orange', 'magenta', 'black', 'indigo']
    for i, df in enumerate(dfs):
        scatter_plot(df, colors[i])
    plt.show()


# Write the info in the dataframe out to json file
def save_dataframe(df):
    # Todo: complete this function
    # Write the data back to the json file (register file)
    pass


# Take the results of the registration and apply them to the original point clouds
def apply_registration_to_pointcloud(filename, angle, offset):
    # Todo: complete this function
    # Open the point cloud, rotate it, offset it, and write it back to file
    basename, ext = filename.split('.')
    outfile = basename + '_registered.' + ext
    pc = PointCloud(filename, render=False)
    pc.rotate_xy(angle)
    pc.translate_xy(offset)
    pc.write(outfile, overwrite=True)


# Take the results of the registration and apply them to the annotations dataframe
def apply_registration_to_dataframe(df, angle, offset):
    rotate_dataframe(df, angle)
    df['points'] += offset


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Please supply reference file and one or more register files')
        sys.exit()

    # Load the annotation data into a list of pandas dataframes
    json_data, shapes, sources, names = [], [], [], []
    for fname in sys.argv[1:]:
        with open(fname) as f:
            data = json.load(f)
            json_data.append(data)
            sources.append(data['sourcePath'])
            names.append(data['roomName'])
            cur_shapes = pd.DataFrame(columns=['label', 'points'])
            for i, s in enumerate(data['shapes']):
                cur_shapes.loc[i] = [s['label'], np.array(s['points'])]
        shapes.append(cur_shapes)

    # Register the rooms (write data to new point cloud files)
    show_points(shapes)
    register(shapes, names, sources)
    show_points(shapes)

    # Write the registered data out to new annotation files
    for i, fname in enumerate(sys.argv[1:]):
        if not i:
            continue
        with open(fname.replace('.json', '_registered.json'), 'w') as f:
            for j in range(len(shapes[i])):
                json_data[i]['shapes'][j]['points'] = shapes[i].loc[j, 'points'].tolist()
            basename, ext = json_data[i]['sourcePath'].split('.')
            outfile = basename + '_registered.' + ext
            json_data[i]['sourcePath'] = outfile
            json.dump(json_data[i], f)
