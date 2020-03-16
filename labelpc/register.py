
import sys
import json
import numpy as np

if len(sys.argv) < 3:
    print('Please supply reference file and register file')

reference_file = sys.argv[1]  # File that defines the reference frame or coordinate system including a door
register_file = sys.argv[2]   # File that is to be registered to the reference file including same door

with open(reference_file) as f:
    reference_data = json.load(f)
with open(register_file) as f:
    register_data = json.load(f)


reference_doors = {}
reference_objects = {}
for s in reference_data['shapes']:
    if 'door' in s['label']:
        reference_doors[s['label']] = np.array(s['points'])
    else:
        reference_objects[s['label']] = np.array(s['points'])

register_doors = {}
register_objects = {}
for s in register_data['shapes']:
    if 'door' in s['label']:
        register_doors[s['label']] = np.array(s['points'])
    else:
        register_objects[s['label']] = np.array(s['points'])

common_doors = {}
for key, value in reference_doors.items():
    if key in register_doors:
        common_doors[key] = value


def distance(p1, p2):
    return np.sqrt(np.dot(p1 - p2, p1 - p2))


def door_loss(common_doors, register_doors):
    total_dist = 0.0
    for label in common_doors.keys():
        total_dist += single_door_loss(label, common_doors, register_doors)
    return total_dist


def single_door_loss(door, reference_doors, register_doors):
    ref1, ref2 = reference_doors[door]
    reg1, reg2 = register_doors[door]
    dist1 = min(distance(ref1, reg1), distance(ref1, reg2))
    dist2 = min(distance(ref2, reg1), distance(ref2, reg2))
    return dist1 + dist2


def non_door_score(reference_objects, register_objects):
    total_dist = 0.0
    for ref_points in reference_objects.values():
        for ref_point in ref_points:
            min_distance = 10000.0
            for reg_points in register_objects.values():
                for reg_point in reg_points:
                    if distance(reg_point, ref_point) < min_distance:
                        min_distance = distance(reg_point, ref_point)
            total_dist += min_distance
    return total_dist


def align_door(label, reference_doors, register_doors, register_objects):
    ref = np.average(reference_doors[label], axis=0)
    reg = np.average(register_doors[label], axis=0)
    offset = ref - reg
    for label in register_doors.keys():
        for i in range(len(register_doors[label])):
            register_doors[label][i] += offset
    for label in register_objects.keys():
        for i in range(len(register_objects[label])):
            register_objects[label][i] += offset
    return ref


def rotate_registered(angle, center, register_doors, register_objects):
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array(((c, s), (-s, c)))
    for label in register_doors.keys():
        for i in range(len(register_doors[label])):
            register_doors[label][i] -= center
            register_doors[label][i] = np.dot(register_doors[label][i], rot)
            register_doors[label][i] += center
    for label in register_objects.keys():
        for i in range(len(register_objects[label])):
            register_objects[label][i] -= center
            register_objects[label][i] = np.dot(register_objects[label][i], rot)
            register_objects[label][i] += center


center = align_door('door1', reference_doors, register_doors, register_objects)
max_loss_angle, max_score_angle = 0.0, 0.0
max_loss, max_score = 0.0, 0.0
for angle in range(20):
    loss = door_loss(common_doors, register_doors)
    score = non_door_score(reference_objects, register_objects)
    print(angle, 20 * angle, loss, score)
    if loss > max_loss:
        max_loss = loss
        max_loss_angle = angle * 20
    if score > max_score:
        max_score = score
        max_score_angle = angle * 20
    rotate_registered(20, center, register_doors, register_objects)

print(' ')
print(max_loss_angle, max_score_angle)
