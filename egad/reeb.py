import jpype
import jpype.imports

import subprocess
import numpy as np

from pathlib import Path


class ReebGraph:
    def __init__(self, jsrc_path):
        """
        :param jsrc_path: (str) Path that contains compiled reeb_graph java project
                                (https://github.com/dbespalov/reeb_graph)
        """
        self.jsrc_path = jsrc_path

        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jsrc_path], convertStrings=True)
        elif not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()

        # These imports are activated by jpype after starting the JVM
        from java.lang import System
        from java.io import PrintStream, File
        # Disable java output.
        System.setOut(PrintStream(File('/dev/null')))  # NUL for windows, /dev/null for unix

        self.erg = jpype.JClass('ExtractReebGraph')()
        self.crg = jpype.JClass('CompareReebGraph')()

        # Set defaults
        self.params = ['4000', '0.005', str(2 ** 7), '0.5']
        self.erg.main(self.params[:3])
        self.crg.main(self.params)
        try:
            (Path.cwd() / 'log_{}_{}_{}_{}'.format(*self.params)).unlink()
        except FileNotFoundError:
            pass

    def extract_reeb_graph(self, mesh_file):
        """
        Extract the reeb graph for a .wrl file.
        Creates a .mrg file with the same name.
        This only needs to be run once per .wrl.
        :param mesh_file: *.wrl mesh from which to extract reeb graph.
        """
        self.erg.main_one(mesh_file)

    def extract_reeb_graph_with_timeout(self, mesh_file, timeout=30.0):
        subprocess.run(
            ['java', 'ExtractReebGraph', self.params[0], self.params[1], self.params[2], mesh_file],
            cwd=self.jsrc_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout
        )

    def compare_reeb_graph(self, mesh_files):
        mat = np.zeros((len(mesh_files), len(mesh_files)))
        for idx1, mf1 in enumerate(mesh_files[:-1]):
            for idx2, mf2 in enumerate(mesh_files[idx1:]):
                self.crg.main_one([mf1, mf2])
                print(mf1.split('/')[-1], mf2.split('/')[-1], '{:0.3f}'.format(self.crg.SIM_R_S))
                v = round(self.crg.SIM_R_S, 3)
                # v = round(max((v - 0.5), 0)/0.5, 3)
                mat[idx1, idx1+idx2] = v
                mat[idx1+idx2, idx1] = v
        return mat

    def compare_two_reeb_graph(self, mf1, mf2):
        self.crg.main_one([mf1, mf2])
        # print(mf1.split('/')[-1], mf2.split('/')[-1], '{:0.3f}'.format(self.crg.SIM_R_S))
        v = round(self.crg.SIM_R_S, 3)
        return v

    @staticmethod
    def trimesh_to_vrml(mesh, output_file):
        """
        Export a trimesh.Trimesh to a VRML file.
        The reeb_graph project requires these in a specifc format.
        TODO: Link this into the trimesh export handlers.
        TODO: Fix the java to be less bad.
        :param mesh: trimesh.Trimesh
        :param output_file: *.wrl output file.
        """

        with open(output_file, 'w') as f:
            f.write('#VRML V2.0 utf8\n')
            f.write('Transform {\n'
                    '   children [\n'
                    '      Shape {\n'
                    '         geometry IndexedFaceSet {\n'
                    '            coord Coordinate {\n'
                    '               point[\n')
            for v in mesh.vertices:
                f.write('                  {:0.6f} {:0.6f} {:0.6f},\n'.format(*v.tolist()))
            f.write('               ] # end of points\n'
                    '            } # end of Coordinate\n'
                    '            coordIndex [\n')

            for face in mesh.faces:
                f.write('               {:d}, {:d}, {:d}, -1,\n'.format(*face.tolist()))

            f.write('            ] # end of coordIndex\n'
                    '         } # end of geometry\n'
                    '         appearance Appearance {\n'
                    '            material Material {\n'
                    '               diffuseColor 0.7 0.7 0.7\n'
                    '               emissiveColor 0.05 0.05 0.05\n'
                    '               specularColor 1.0 1.0 1.0\n'
                    '               ambientIntensity 0.2\n'
                    '               shininess 0.2\n'
                    '               transparency 0.0\n'
                    '            } #end of material\n'
                    '         } #end of appearance\n'
                    '      } # end of Shape\n'
                    '   ] # end of children\n'
                    '}\n')
