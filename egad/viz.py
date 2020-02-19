import copy
import numpy as np
import imageio

import trimesh, trimesh.smoothing, trimesh.path, trimesh.scene, trimesh.creation

import graphviz


def plot_mesh(mesh):
    sc = trimesh.scene.scene.Scene()
    sc.add_geometry(mesh)

    marker = trimesh.creation.axis(origin_size=0.01, axis_radius=0.01, axis_length=1)
    sc.add_geometry(marker)

    sc.set_camera(angles=(np.pi/4, 0, np.pi/4), distance=2, center=mesh.center_mass)

    sc.show()


def get_mesh_image(mesh, resolution=(640, 480)):
    sc = trimesh.scene.scene.Scene()
    sc.add_geometry(mesh)
    sc.set_camera(angles=(np.pi / 4, 0, np.pi / 4), distance=0.2, center=mesh.center_mass)
    im = imageio.imread(sc.save_image(resolution=resolution))
    t = (im[:,:,:3].min(axis=2) != 255).astype(np.uint8) * 255
    im[:,:,3] = t
    return im


def plot_vox_grid(voxgrid):
    if np.sum(voxgrid) == 0:
        return

    vg = trimesh.voxel.VoxelGrid(voxgrid)
    m = vg.marching_cubes
    m = sorted(m.split().tolist(), key=lambda x: x.volume)[-1]

    m.vertices /= np.array(voxgrid.shape)
    plot_mesh(m)


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=True,
             node_colors=None, fmt='pdf'):
    """ Receives a genome and draws a neural network with arbitrary topology. """

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    def node_to_string(n):
        try:
            no = genome.nodes[n]
            return '\n'.join([str(n), str(no.activation), '{:0.2f}'.format(no.bias)])
        except KeyError:
            return str(n)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, node_to_string(k))
        input_attrs = {'style': 'filled', 'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, node_to_string(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
        dot.node(name, _attributes=node_attrs)

    from neat.graphs import required_for_output
    required = required_for_output(config.genome_config.input_keys, config.genome_config.output_keys, genome.connections)
    required |= inputs
    # print(required)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            # print(cg)
            f, t = cg.key
            if f not in required or t not in required:
                # print('removed')
                continue
            if not cg.enabled and not show_disabled:
                # print('remove')
                continue
            # print('kept')
            connections.add(cg.key)

        # for c in connections:
        #     print(c)

        used_nodes = inputs | outputs
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in used_nodes and b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            # print(pending, new_pending)
            pending = new_pending
        # used_nodes = required
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled'}
        attrs['fillcolor'] = node_colors.get(n, 'white')
        dot.node(node_to_string(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            if input not in used_nodes or output not in used_nodes:
                continue
            a = node_names.get(input, node_to_string(input))
            b = node_names.get(output, node_to_string(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width, 'label': '{:0.2f}'.format(cg.weight), 'fontsize': '9'})

    dot.render(filename, view=view, cleanup=True)

    return dot