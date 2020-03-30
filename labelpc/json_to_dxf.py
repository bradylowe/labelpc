
import json
import ezdxf
import sys


def get_rectangle_scale(points):
    x, y = points[1][0] - points[0][0], points[1][1] - points[0][1]
    return max(x, y), min(x, y)


def get_rectangle_rotation(orientation):
    return int(90 * orientation)


def get_rectangle_center(points):
    return (points[0][0] + points[1][0]) / 2.0, (points[0][1] + points[1][1]) / 2.0


if __name__ == "__main__":

    # Get the data from the json file
    if len(sys.argv) < 2:
        print('Please supply json file')
        sys.exit()
    json_file = sys.argv[1]  # File that defines the reference frame or coordinate system including a door
    with open(json_file) as f:
        json_data = json.load(f)

    dxf = ezdxf.new(dxfversion='R2010')
    msp = dxf.modelspace()
    rectangle = dxf.blocks.new(name='RECTANGLE')
    rectangle.add_lwpolyline([(-0.5,-0.5), (0.5,-0.5), (0.5,0.5), (-0.5,0.5), (-0.5,-0.5)])
    ibeam = dxf.blocks.new(name='IBEAM')
    ibeam.add_line((-0.5, -0.5), (0.5, -0.5))
    ibeam.add_line((-0.5, 0.5), (0.5, 0.5))
    ibeam.add_line((0, -0.5), (0, 0.5))
    beam_scale = 0.1
    dxf.layers.new('WALLS', dxfattribs={'color': 5})
    dxf.layers.new('PALLETS', dxfattribs={'color': 1})
    dxf.layers.new('RACKS', dxfattribs={'color': 6})
    dxf.layers.new('DOORS', dxfattribs={'color': 3})
    dxf.layers.new('BEAMS', dxfattribs={'color': 2})
    dxf.layers.new('I-BEAMS', dxfattribs={'color': 4})

    for shape in json_data['shapes']:
        if shape['label'] == 'walls':
            points = shape['points']
            points.append(shape['points'][0])
            walls = dxf.blocks.new(name='WALLS')
            walls.add_lwpolyline(points)
            msp.add_blockref('WALLS', (0, 0), dxfattribs={'layer': 'WALLS'})
        elif shape['label'] == 'pallet' or 'rack' in shape['label']:
            x_scale, y_scale = get_rectangle_scale(shape['points'])
            rotation = get_rectangle_rotation(shape['orient'])
            layer = 'PALLETS' if shape['label'] == 'pallet' else 'RACKS'
            msp.add_blockref('RECTANGLE', get_rectangle_center(shape['points']), dxfattribs={
                'xscale': x_scale,
                'yscale': y_scale,
                'rotation': rotation,
                'layer': layer
            })
        elif 'door' in shape['label']:
            msp.add_line(shape['points'][0], shape['points'][1], dxfattribs={
                'layer': 'DOORS'
            })
        elif shape['label'] == 'beam':
            msp.add_blockref('RECTANGLE', shape['points'][0], dxfattribs={
                'layer': 'BEAMS',
                'xscale': beam_scale,
                'yscale': beam_scale
            })
        elif shape['label'] == 'I_beam':
            msp.add_blockref('IBEAM', shape['points'][0], dxfattribs={
                'layer': 'I-BEAMS',
                'xscale': beam_scale,
                'yscale': beam_scale
            })
        else:
            print('Unsupported shape type')

    dxf.saveas(json_file.replace('.json', '.dxf'))
