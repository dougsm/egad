"""
Generate the EGAD dataset using CPPN-NEAT.

Usage:

python generate_dataset.py <path to config file.json> <--resume>

"""

import argparse
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import sys
import time
from pathlib import Path

import subprocess
import imageio
import numpy as np
import trimesh
import pickle
from scipy.ndimage import binary_opening
from trimesh.voxel.ops import fill_orthographic

from itertools import repeat

from egad import cppn_neat
from egad.mesh import smooth_mesh, voxel_to_mesh, scale_mesh
from egad.viz import get_mesh_image
from egad.reeb import ReebGraph

ctx._force_start_method('spawn')  # Required for the JVM (ReebGraph)


RG = None  # ReebGraph Interface
DN = None  # Dexnet command


def pool_init(dexnet_path, reeb_path):
    """
    Initialiser for spawned processes
    """
    global RG, DN
    DN = dexnet_path
    RG = ReebGraph(reeb_path)


def pool_compare(args):
    """
    Process a new mesh:
    1. Save to disk.
    2. Compute complexity.
    3. Compute grasp difficulty.
    4. Compute similarity to every other mesh currently in the pool. (and diversity)

    Return: diversity, complexity, difficulty, list of similarities for updating cache.

    """
    global RG, DN
    try:
        m_path, pool, cutoff, k_neighbours = args

        m_idx, m_path = m_path
        m_path = Path(m_path)

        # Create and save the mesh
        try:
            m = CPPNMesh(m_path.parent, m_idx, mesh=trimesh.load(m_path))
            if DN == 'FAKE':
                dexnet_return = np.random.randint(2, 102)  # Fake random number
            else:
                # Value is encoded in the returncode from the interface script.
                dexnet_return = subprocess.run(DN + ' dexnet_grasp_rank_interface.py ' + str(m_path.resolve()),
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
                print(dexnet_return)
            if dexnet_return in [0, 1]:
                return 0.0, -1, -2, []
            dexnet_return -= 2  # Shifted up by 2 to avoid overlap with exit 1 which is a legit error
        except (subprocess.CalledProcessError, RuntimeError):
            diversity = 0.0
            return diversity, -1, -1, []

        if not pool:
            # No pool exists, so no comparisons to make
            diversity = 1.0
            return diversity, m.complexity, dexnet_return, []

        dists = []
        dists_all = []

        # Compute similarity to every other mesh in the pool.
        # Keep a list of closest neighbours (dists) for diversity calculation.
        for pm in pool:
            t0 = time.process_time()
            d = 1.0 - RG.compare_two_reeb_graph(m.vrml_path, pm)
            t1 = time.process_time()
            if (t1 - t0) > 1.0:
                # If it takes this long, something is broken. Bail.
                return 0.0, -1, -3, []
            dists_all.append(d)
            if len(dists) >= k_neighbours:
                if d < dists[-1]:
                    dists[-1] = d
                    dists.sort()
                    # Early stopping if less than worse current.
                    if np.mean(dists) < cutoff:
                        break
            else:
                dists.append(d)
                dists.sort()

        diversity = np.mean(dists)
        return diversity, m.complexity, dexnet_return, dists_all

    except Exception as e:
        # print(str(e))
        return 0.0, -1, -4, []


def evaluate_new_meshes(mesh_pool, new_mesh_paths, cutoff, k_neighbours, args):
    """
    Spawn up a multiprocessing pool to process the new workers.
    :param mesh_pool:  Current mesh pool (i.e. search space)
    :param new_mesh_paths: Paths to the new mesh files.
    :param cutoff: Lower cutoff on diversity.
    :param k_neighbours: Number of neighbours to consider for diversity.
    :param config_dict: config file
    :return: list of return values from pool_compare, list of mesh ids for indexing.
    """
    mpvs = list(mesh_pool.values())

    pool_mesh_paths = [m.vrml_path for ms in mpvs for m in ms]
    pool_mesh_ids = [m.idx for ms in mpvs for m in ms]

    l = []
    dexnet_path = args.dexnetenv
    reeb_path = args.reebpath
    print('Starting processing pool with {} workers'.format(args.processes))
    with mp.Pool(args.processes, initializer=pool_init, initargs=[dexnet_path, reeb_path]) as p:
        res = p.imap(pool_compare,
                     zip(new_mesh_paths, repeat(pool_mesh_paths), repeat(cutoff), repeat(k_neighbours)),
                     chunksize=args.chunksize)

        for i, r in enumerate(res):
            l.append(r)
            print('{:d}/{:d}'.format(i+1, len(new_mesh_paths)), 'Div: {:0.3f}, Cplx: {:0.3f}, Diff: {:0.2f}'.format(r[0], r[1], r[2]/100.0))

        p.close()
        p.join()
    return l, pool_mesh_ids


class CPPNMesh:
    """
    Class to represent and process a mesh output by a CPPN.
    """
    def __init__(self, output_dir, idx, mesh=None, genome=None, genome_idx=None, load_genome=False):
        """
        :param output_dir: Directory to save to/load from
        :param idx: Unique id of this mesh
        :param mesh: Trimesh mesh object (optional -- if provided, will be processed and saved)
        :param genome: CPPN Genome (optional)
        :param genome_idx:
        :param load_genome:
        """
        self.output_path = Path(output_dir) / '{}'.format(idx)
        self.idx = idx
        self._diversity = 0.0
        self._diversity_neighbours = set()
        self.complexity = 0
        self.genome = genome
        self.genome_idx = genome_idx

        if mesh is not None:
            self.complexity = CPPNMesh.morphological_complexity(mesh)
            self.save(mesh)

        if load_genome:
            self.load_genome()

    def __lt__(self, other):
        return self.diversity < other.diversity

    @property
    def diversity(self):
        return self._diversity

    @diversity.setter
    def diversity(self, val):
        self._diversity = val
        if self.genome:
            self.genome.fitness = val

    @property
    def mesh_path(self):
        return self.output_path.with_suffix('.obj')

    @property
    def vrml_path(self):
        return str(self.output_path.with_suffix('.wrl'))

    @property
    def thumb_path(self):
        return self.output_path.with_suffix('.png')

    @property
    def genome_path(self):
        return self.output_path.with_suffix('.pickle')

    def save_genome(self):
        if self.genome_path.exists():
            return
        with open(self.genome_path, 'wb') as f:
            data = (self.genome_idx, self.genome)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_genome(self):
        with open(self.genome_path, 'rb') as f:
            self.genome_idx, self.genome = pickle.load(f)

    def save(self, mesh):
        global RG
        RG.trimesh_to_vrml(mesh, self.vrml_path)
        RG.extract_reeb_graph_with_timeout(self.vrml_path)
        imageio.imsave(self.thumb_path, get_mesh_image(mesh, resolution=(64, 64)))

    @staticmethod
    def morphological_complexity(mesh):
        """
        Measure of morphological complexity from

        Environmental Influence on the Evolution of Morphological Complexity in Machines
        Joshua E. Auerbach, and Josh C. Bongard
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3879106/

        Given a mesh:
        1. Computes the angular deficit of the triangles at each vertex
        2. Compuates a histogram over the angular deficit
        3. Returns the entropy of the normalised histogram

        """

        # Compute internal angles per vertex
        flat_vertices = mesh.faces.flatten()
        flat_angles = mesh.face_angles.flatten()
        unique_vertices = np.unique(flat_vertices)
        vertex_angles = np.array([2*np.pi - flat_angles[flat_vertices == v_id].sum() for v_id in unique_vertices])

        # Create Normalised Histogram
        hist = np.histogram(vertex_angles, bins=512, range=(-np.pi*2, np.pi*2))[0].astype(np.float)
        hist /= hist.sum()

        # Compute entropy
        H = -1 * (hist * np.log2(hist+1e-6)).sum()
        H = max(0, H)
        return H


class DiversityExperiment:
    def __init__(self, args):
        self.args = args

        self.DIV_THRESH = args.divthreshold
        self.K_NEIGHBOURS = args.neighbours
        self.MAX_LEN = args.cellsize

        self._pool_last_changed = None

        if args.resume:
            # Attempt to find the state of the last run in order to resume.
            p = sorted(args.outputdir.glob('*'))[-1]
            p = sorted(p.glob('checkpoints/generation_*'))
            gen_ids = [(int(str(g).split('_')[-1]), g) for g in p]
            gen_ids.sort()
            p = gen_ids[-1][1]

            self.generation = int(str(p).split('_')[-1])

            checkpoint = sorted(p.glob('checkpoint*'))[-1]
            print('Loading from {}'.format(checkpoint))
        else:
            checkpoint = None
            self.generation = 0

        # Initialise the NEAT algorithm.
        self.inputs = cppn_neat.get_workspace(args.cppnsize, args.cppnsize, args.cppnsize)
        self.neat = cppn_neat.NEAT(self.inputs, checkpoint=checkpoint,
                                   output_dir=str(args.outputdir),
                                   config=args.cppnconfig)

        self.output_dir = Path(self.neat.output_dir)
        self.meshpool_dir = self.output_dir / 'pool'
        self.meshpool_dir.mkdir(parents=True, exist_ok=True)
        self.div_output_dir = self.output_dir / 'diversity'
        self.div_output_dir.mkdir(parents=True, exist_ok=True)

        # Try to load existing similarity cache.
        if args.resume and (self.output_dir / 'sim_cache.json').exists():
            try:
                with open((self.output_dir / 'sim_cache.json')) as f:
                    self.sim_cache = json.load(f)
            except:
                self.sim_cache = {}
        else:
            self.sim_cache = {}

        self.mesh_pool = {}

        if args.resume:
            with open(self.output_dir / 'gen_{:04d}.json'.format(self.generation)) as f:
                mp_ids = json.load(f)
                print(mp_ids)
            for m_key in mp_ids:
                print('Loading {}'.format(m_key))
                try:
                    self.mesh_pool[m_key] = [CPPNMesh(self.meshpool_dir, mfp, load_genome=True) for mfp in mp_ids[m_key]]
                except:
                    print('problem loading')
                    raise
            self.rerank_mesh_pool()

    def get_similarity(self, m1, m2):
        """
        Get the similarity between two meshes, either from the cache or using reeb graphs
        :param m1:  Mesh 1
        :param m2:  Mesh 2
        :return:  Similarity between two meshes
        """
        k = str((min(m1.idx, m2.idx), max(m1.idx, m2.idx)))  # make it json safe
        if k in self.sim_cache:
            return self.sim_cache[k]
        else:
            t0 = time.time()
            v = RG.compare_two_reeb_graph(m1.vrml_path, m2.vrml_path)
            t1 = time.time()
            if t1 - t0 > 1.0:
                print('Similarity Timeout {}, {}, {:0.3f}'.format(m1.idx, m2.idx, (t1-t0)))
                raise ValueError()
            self.sim_cache[k] = v
            return v

    def get_set_mesh_diversity(self, m, pool, cutoff=0.):
        """
        Copmare a mesh to the entire pool to compute its diversity.
        :param m: Mesh to compare
        :param pool: Pool to compare against
        :param cutoff: Minimum diversity at which to stop checking (less accurate but faster).
        :return: Diversity
        """
        if not pool:
            m.diversity = 1.0
            return 1.0

        dists = []
        neighbours = []

        # Iterate the pool
        for pms in pool.values():
            for pm in pms:
                if m.idx == pm.idx:
                    continue

                # Get similarity
                try:
                    d = 1.0 - self.get_similarity(pm, m)
                except ValueError:
                    m.diversity = 0.0
                    m._diversity_neighbours = set()
                    return 0.0

                # Keep K_NEIGHBOURS closest neighbours
                if len(dists) >= self.K_NEIGHBOURS:
                    if d <= dists[-1]:
                        dists[-1] = d
                        dists.sort()

                        neighbours[-1] = (d, pm.idx)
                        neighbours.sort(key=lambda x: x[0])
                        # Early stopping if less than cutoff (can't get better, only worse).
                        if np.mean(dists) < cutoff:
                            break
                else:
                    dists.append(d)
                    dists.sort()
                    neighbours.append((d, pm.idx))
                    neighbours.sort(key=lambda x: x[0])

        # Compute diversity
        diversity = np.mean(dists)
        m.diversity = diversity
        m._diversity_neighbours = set([n[1] for n in neighbours])
        return diversity

    def rerank_mesh_pool(self):
        """
        Re-compute the diversity of every mesh in the pool.
        :return: The cell with the lowest minimum diversity.
        """
        min_div = 1.0
        min_ms = None
        for ms in self.mesh_pool.values():
            for m in ms:
                if self._pool_last_changed is None or self._pool_last_changed in m._diversity_neighbours:
                    self.get_set_mesh_diversity(m, self.mesh_pool)
            if len(ms) > self.MAX_LEN:
                ms.sort(reverse=True)
                if ms[-1].diversity < min_div:
                    min_div = ms[-1].diversity
                    min_ms = ms
        return min_ms

    def save_sim_cache(self):
        """
        Save the similarity cache to disk.
        """
        new_sim_cache = {}
        mp_flat = [m for ms in self.mesh_pool.values() for m in ms]
        for idx, m1 in enumerate(mp_flat):
            for m2 in mp_flat[idx:]:
                k = str((min(m1.idx, m2.idx), max(m1.idx, m2.idx)))
                if k in self.sim_cache:
                    new_sim_cache[k] = self.sim_cache[k]
        self.sim_cache = new_sim_cache
        with open(self.output_dir / 'sim_cache.json', 'w') as f:
            json.dump(self.sim_cache, f)

    def _process_voxgrid(self, voxgrid):
        # Fill high-frequency holes
        voxgrid = fill_orthographic(voxgrid)
        voxgrid = binary_opening(voxgrid)
        mesh = voxel_to_mesh(voxgrid)
        smooth_mesh(mesh)
        mesh = scale_mesh(mesh, 1 / 250)  # Rescale to 0.1m
        # Min size.
        max_dim = max(list(mesh.bounding_box.primitive.extents))
        scale = 0.1 / max_dim
        scale_mesh(mesh, scale)
        return mesh

    def _genomes_to_meshes(self, config, genomes):
        new_mesh_paths = []
        kept_genomes = []
        # Process the individual genomes
        for gidx, g in genomes:
            idx = '{:04d}_{:04d}'.format(self.generation, gidx)
            voxgrid = cppn_neat.CPPN(g, config, self.inputs).query()

            # Check for 0 volume.
            if voxgrid.sum() == 0:
                g.fitness = 0.0
                print('{}: Ignored, no volume'.format(idx))
                continue

            # Pre-process the mesh.
            try:
                mesh = self._process_voxgrid(voxgrid)
            except:
                print('{}: Ignored, exception occured'.format(idx))
                g.fitness = 0.0
                continue

            # Export the mesh.
            new_mesh_p = (self.meshpool_dir / idx).with_suffix('.uf.obj')
            mesh.export(new_mesh_p)

            new_mesh_paths.append((idx, str(new_mesh_p)))
            kept_genomes.append((gidx, g))

            print('{}: Created Successfully'.format(idx))
        return kept_genomes, new_mesh_paths

    def _update_pool_new_meshes(self, kept_genomes, new_mesh_paths, pool_mesh_ids, res):
        new_meshes = []
        for (nmidx, nmp), (gidx, g), (diversity, complexity, difficulty, similarities) in zip(new_mesh_paths,
                                                                                              kept_genomes, res):
            m = CPPNMesh(self.meshpool_dir, nmidx, genome=g, genome_idx=gidx)
            m.diversity = diversity
            g.fitness = diversity

            # Update the cache with results from each new mesh.
            for mp_m, mp_sim in zip(pool_mesh_ids, similarities):
                k = str((min(mp_m, nmidx), max(mp_m, nmidx)))
                self.sim_cache[k] = 1.0 - mp_sim

            print('Mesh: {}, Diversity: {:0.2f}'.format(nmidx, diversity))

            # Find cell in the map and add if possible.
            if diversity > self.DIV_THRESH:
                complexity = int(np.clip(complexity - 1.0, 0.0, 4.0 - 1e-6) / 4.0 * self.args.complexitydims)
                difficulty = int(np.clip(difficulty, 0, 99) / 100 * self.args.difficultydims)
                k = str((complexity, difficulty))
                if k not in self.mesh_pool:
                    self.mesh_pool[k] = [m]
                    new_meshes.append(m)
                elif diversity >= self.mesh_pool[k][-1].diversity:
                    self.mesh_pool[k].append(m)
                    self.mesh_pool[k].sort(reverse=True)
                    new_meshes.append(m)
        if new_meshes:
            new_meshes.sort(reverse=True)
            print('Max Diversity New: {:0.2f}'.format(new_meshes[0].diversity))
            print('Min Diversity New: {:0.2f}'.format(new_meshes[-1].diversity))
            print('            N New: {:d}'.format(len(new_meshes)))

    def evaluate_genomes(self, genomes, config):
        """
        Callback for python-neat.
        Process a population of CPPNs.
        """
        kept_genomes, new_mesh_paths = self._genomes_to_meshes(config, genomes)

        # Evaluate the viable meshes in parallel.
        res, pool_mesh_ids = evaluate_new_meshes(self.mesh_pool, new_mesh_paths, self.DIV_THRESH, self.K_NEIGHBOURS, self.args)

        # Add new meshes to the pool
        self._update_pool_new_meshes(kept_genomes, new_mesh_paths, pool_mesh_ids, res)

        print('Re-ranking Mesh Pool')
        self._pool_last_changed = None
        ms = self.rerank_mesh_pool()  # Returns the least diverse with multiple in a cell
        while ms:
            self._pool_last_changed = ms.pop(-1).idx
            ms = self.rerank_mesh_pool()

        all_flat = sorted([m.diversity for ms in self.mesh_pool.values() for m in ms], reverse=True)
        print('Max Diversity Pool: {:0.2f}'.format(all_flat[0]))
        print('Min Diversity Pool: {:0.2f}'.format(all_flat[-1]))
        print('         Pool Size: {:d}'.format(len(all_flat)))

        print('Saving current pool')
        curr_gen_ids = {k: [m.idx for m in ms] for k, ms in self.mesh_pool.items()}
        with open(self.output_dir / 'gen_{:04d}.json'.format(self.generation), 'w') as f:
            json.dump(curr_gen_ids, f)

        print('Saving Diversities')
        mesh_diversities = {m.idx: m.diversity for ms in self.mesh_pool.values() for m in ms}
        with open(self.div_output_dir / 'gen_div_{:04d}.json'.format(self.generation), 'w') as f:
            json.dump(mesh_diversities, f)

        print('Saving Similarity Cache')
        self.save_sim_cache()

        self.generation += 1
        print('Generation Done')

        # Re-sample the genomes from the map.
        [m.save_genome() for ms in self.mesh_pool.values() for m in ms]
        pop_genomes = {m.genome_idx: m.genome for ms in self.mesh_pool.values() for m in ms}
        pop_genomes.update(self.neat._pop.population)  # Keep the original genomes.
        self.neat._pop.population = pop_genomes
        self.neat._pop.species.speciate(None, pop_genomes, self.neat._pop.generation)

    def run(self):
        self.neat.run(2000, self.evaluate_genomes)


def run():
    global RG

    parser = argparse.ArgumentParser(description='Generate shape dataset using CPPNs.')

    # CPPN
    parser.add_argument('--cppnsize', type=int, default=25, help='Dimension of CPPN voxel grid.')
    parser.add_argument('--cppnconfig', type=Path, default=Path('cfg/config_cppn_voxgrids_diversity'),
                        help='path to CPPN config file')

    # MAP
    parser.add_argument('--mapsize', type=int, default=25, help='Dimension of the search space')
    parser.add_argument('--complexitydims', type=int, default=25, help='Number of cells in complexity dimension')
    parser.add_argument('--difficultydims', type=int, default=25, help='Number of cells in difficulty dimension')
    parser.add_argument('--cellsize', type=int, default=4, help='Meshes to keep per cell')
    parser.add_argument('--neighbours', type=int, default=10, help='Number of neighbours to compute diversity')
    parser.add_argument('--divthreshold', type=float, default=0.0, help='Minimum threshold for object diversity')

    # Multiprocessing
    parser.add_argument('--processes', type=int, default=4, help='Number of procesing threads')
    parser.add_argument('--chunksize', type=int, default=1, help='Number of jobs per thread at a time')

    # External dependencies
    parser.add_argument('--reebpath', type=Path, default=Path('/home/co/reeb/reeb_graph/src'), help='Path to reeb graph implementation')
    parser.add_argument('--dexnetenv', type=str, default='python2', help='Python environment for dexnet')

    # Other
    parser.add_argument('--outputdir', type=Path, default=Path('/output'))
    parser.add_argument('--resume', action='store_true', help='resume from last run')
    args = parser.parse_args()

    args.reebpath = str(args.reebpath.resolve())
    RG = ReebGraph(args.reebpath)

    DiversityExperiment(args).run()


if __name__ == '__main__':
    run()