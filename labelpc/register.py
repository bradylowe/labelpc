
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Please supply reference file and register file')
    sys.exit()

reference_file = sys.argv[1]  # File that defines the reference frame or coordinate system including a door
register_file = sys.argv[2]   # File that is to be registered to the reference file including same door

with open(reference_file) as f:
    reference_data = json.load(f)
with open(register_file) as f:
    register_data = json.load(f)


ref_shapes = pd.DataFrame(columns=['label', 'points'])
for i, s in enumerate(reference_data['shapes']):
    ref_shapes.loc[i] = [s['label'], np.array(s['points'])]
reg_shapes = pd.DataFrame(columns=['label', 'points'])
for i, s in enumerate(register_data['shapes']):
    reg_shapes.loc[i] = [s['label'], np.array(s['points'])]


def find_in_dataframe(df, query='door', indices=False):
    if indices:
        return df.index[df['label'].str.contains(query)].tolist()
    else:
        return df['label'].str.contains(query)


# Doors must have common name theme. (DOOR before DOCK, DOOR1 before DOOR2)
def find_common_doors(ref, reg):
    common = []
    ref_doors = ref.loc[ref['label'].str.contains('door')]
    for label in ref_doors['label'].unique():
        doors = reg.loc[reg['label'].str.contains(label)]
        if len(doors):
            common.append(label)
    return common


def distance(p1, p2):
    return np.sqrt(np.dot(p1 - p2, p1 - p2))


def door_loss(door, ref, reg):
    total = 0.0
    for ref_points in ref.loc[ref['label'] == door, 'points']:
        min_dist = 100000.0
        for reg_points in reg.loc[reg['label'] == door, 'points']:
            ref_c, reg_c = ref_points.mean(axis=0), reg_points.mean(axis=0)
            min_dist = min(min_dist, distance(ref_c, reg_c))
        total += min_dist
    return total


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


def iou(df1, df2):
    boxA = dataframe_bounds(df1)
    boxB = dataframe_bounds(df2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def register(door, ref, reg):
    best_angle, best_offset, min_dist = None, None, 1000000.0
    for i in range(4):
        for ref_points in ref.loc[ref['label'] == door, 'points']:
            for reg_points in reg.loc[reg['label'] == door, 'points']:
                cur_reg = reg.copy()
                offset = ref_points.mean(axis=0) - reg_points.mean(axis=0)
                cur_reg['points'] += offset
                dist = door_loss(door, ref, cur_reg) * (1.0 + iou(ref, cur_reg))
                if dist < min_dist:
                    min_dist = dist
                    best_offset = offset
                    best_angle = i * 90.0
        rotate_dataframe(reg, 90.0)

    rotate_dataframe(reg, best_angle)
    reg['points'] += best_offset


def scatter_plot(df, marker='.'):
    x, y, c = [], [], []
    for idx, row in df.iterrows():
        for point in row['points']:
            x.append(point[0]), y.append(point[1])
            if 'door' in row['label']:
                c.append('red')
            else:
                c.append('blue')
    plt.scatter(x, y, c=c, marker=marker)


def show_points(df1, df2):
    scatter_plot(df1, '.')
    scatter_plot(df2, 'X')
    plt.show()


print(door_loss(find_common_doors(ref_shapes, reg_shapes)[0], ref_shapes, reg_shapes))
show_points(ref_shapes, reg_shapes)
register('door_room5_room6', ref_shapes, reg_shapes)
show_points(ref_shapes, reg_shapes)
print(door_loss(find_common_doors(ref_shapes, reg_shapes)[0], ref_shapes, reg_shapes))
