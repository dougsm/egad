import os.path as osp
import time
from collections import OrderedDict
from os import makedirs
from pathlib import Path

import neat
import numpy as np
import torch
from progressbar import ProgressBar, Bar, AdaptiveETA, Percentage, SimpleProgress, FileTransferSpeed
from pytorch_neat.cppn import create_cppn

from .mesh import voxel_to_mesh, scale_mesh, smooth_mesh
from .reproduction import MAPReproduction
from .viz import draw_net


def _scale_workspace(t):
    return (t - t.min()) / (t.max() - t.min()) * 2.0 - 1.0


def get_workspace(x_vox, y_vox, z_vox):
    x, y, z = np.indices((x_vox, y_vox, z_vox))

    input_tensor_x = _scale_workspace(torch.tensor(x, dtype=torch.float))
    input_tensor_y = _scale_workspace(torch.tensor(y, dtype=torch.float))
    input_tensor_z = _scale_workspace(torch.tensor(z, dtype=torch.float))

    input_tensor_rxy = torch.sqrt(input_tensor_x**2 + input_tensor_y**2)
    input_tensor_rxz = torch.sqrt(input_tensor_x**2 + input_tensor_z**2)
    input_tensor_ryz = torch.sqrt(input_tensor_y**2 + input_tensor_z**2)
    input_tensor_rxyz = torch.sqrt(input_tensor_x ** 2 + input_tensor_y ** 2 + input_tensor_z ** 2)

    input_tensor_b = torch.zeros_like(input_tensor_x)

    return OrderedDict([
        ('x', input_tensor_x),
        ('y', input_tensor_y),
        ('z', input_tensor_z),

        ('rxy', input_tensor_rxy),
        ('rxz', input_tensor_rxz),
        ('ryz', input_tensor_ryz),

        ('rxyz', input_tensor_rxyz),

        ('b', input_tensor_b),
        ]
    )


class NoSpeciesSet(neat.species.DefaultSpeciesSet):
    def speciate(self, config, population, generation):
        sid = next(self.indexer)
        self.species = {}
        s = self.species.get(sid)
        if s is None:
            s = neat.species.Species(sid, generation)
            self.species[sid] = s
        self.genome_to_species = {}
        for gid in population:
            self.genome_to_species[gid] = sid
        s.update(None, population)


def load_config(config_file='config_cppn_voxgrids_default'):
    return neat.config.Config(neat.genome.DefaultGenome,
                              MAPReproduction, #neat.reproduction.DefaultReproduction, #ProportionateReproduction,
                              NoSpeciesSet, # neat.species.DefaultSpeciesSet,
                              neat.stagnation.DefaultStagnation,
                              config_file)


class CPPN:
    def __init__(self, genome, config, inputs, threshold=0.5):
        self._cppn = create_cppn(genome, config, inputs.keys(), ['w'])[0]
        self._inputs = inputs
        self._threshold = threshold

    def query(self):
        return self._cppn(**self._inputs).numpy() > self._threshold

    def query_raw(self):
        return self._cppn(**self._inputs).numpy()

    def query_mesh(self):
        return voxel_to_mesh(self.query())


class FitnessFunctionTarget:
    """
    Fitness function for comparison to a target voxel grid.
    """
    def __init__(self, target_vox):
        self.target_vox = target_vox.astype(np.int)

    def __call__(self, test_vox):
        fns = np.sum(test_vox.astype(np.int) == self.target_vox)
        fns /= np.prod(self.target_vox.shape)
        return float(2000 ** fns)


class Progress:
    def __init__(self, generations):
        widgets = ['Generation ', SimpleProgress(), ' (', Percentage(), ') ', Bar(), ' ', AdaptiveETA(), ' ',
                   FileTransferSpeed(unit='Gens')]
        self.pbar = ProgressBar(widgets=widgets, maxval=generations)
        self.current = 1

    def step(self):
        self.pbar.update(self.current)
        self.current += 1

    def start(self):
        self.pbar.start()

    def finish(self):
        self.pbar.finish()


class SaveOutputReporter(neat.reporting.BaseReporter):
    """Definition of the reporter interface expected by ReporterSet."""
    def __init__(self, output_dir, inputs):
        super().__init__()
        self.output_dir = output_dir
        self.inputs = inputs
        self.generation = None

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        for idx, genome in population.items():
            if genome.fitness > 0:
                voxgrid = CPPN(genome, config, self.inputs).query()
                try:
                    mesh = smooth_mesh(scale_mesh(voxel_to_mesh(voxgrid), 1 / 250))
                except:
                    continue

                if not mesh.is_watertight or mesh.volume < 0.02 ** 3:
                    continue

                output_dir = osp.join(
                    self.output_dir,
                    'generation_{:04d}'.format(self.generation)
                )
                makedirs(output_dir, exist_ok=True)

                # Save the mesh.
                output_path = osp.join(
                    output_dir,
                    '{:0.2f}_{:d}.stl'.format(genome.fitness, idx)
                )
                mesh.export(output_path)

                # Save the graph.
                output_path = osp.join(
                    output_dir,
                    '{:0.2f}_{:d}'.format(genome.fitness, idx)
                )
                draw_net(config, genome, view=False, filename=output_path,
                         node_names=None, show_disabled=True,
                         prune_unused=True, node_colors=None, fmt='pdf')


class MyCheckpointer(neat.checkpoint.Checkpointer):
    def __init__(self, generation_interval=100, time_interval_seconds=300, output_dir=''):
        super().__init__(generation_interval, time_interval_seconds)
        self.output_dir = output_dir

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        self.filename_prefix = osp.join(
            self.output_dir,
            'checkpoints',
            'generation_{:04d}'.format(generation),
            'checkpoint'
        )
        Path(self.filename_prefix).parent.mkdir(parents=True, exist_ok=True)
        super().save_checkpoint(config, population, species_set, generation)


class NEAT:
    def __init__(self, inputs, config=None, checkpoint=None, output_dir=None):
        self.inputs = inputs

        if checkpoint:
            self._pop = neat.Checkpointer().restore_checkpoint(checkpoint)
            self.config = self._pop.config
            output_dir = str(Path(checkpoint).parent.parent.parent)
            print('Loaded from {}'.format(checkpoint))
        else:
            if not config:
                fp = osp.dirname(__file__)
                config = fp + '/config_cppn_voxgrids_default'
            self.config = load_config(config)

            # Make input keys length dynamic, so I don't have to keep changing
            #  the config file.
            n = len(inputs.keys())
            self.config.genome_config.num_inputs = n
            self.config.genome_config.input_keys = list(range(-n, 0))

            self._pop = neat.population.Population(self.config)

            if output_dir is None:
                output_dir = '/mnt/storage/gwi_output/learn_easiest/{:d}/'.format(int(time.time()))
            else:
                output_dir = osp.join(output_dir, '{:d}/'.format(int(time.time())))

        self.output_dir = output_dir
        # self._pop.add_reporter(SaveOutputReporter(output_dir, inputs))
        self._pop.add_reporter(neat.reporting.StdOutReporter(True))
        self._pop.add_reporter(MyCheckpointer(generation_interval=1, output_dir=output_dir))

    def run(self, generations, fitness_function):
        winner = self._pop.run(fitness_function, generations)
        return CPPN(winner, self.config, self.inputs)

