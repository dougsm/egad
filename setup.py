from setuptools import setup


setup(name='egad',
      version='0.1',
      description='EGAD: an Evolved Grasping Analysis Dataset',
      url='http://github.com/dougsm/egad',
      author='Doug Morrison',
      install_requires=[
          'neat-python',
          'numpy',
          'scipy',
          'click',
          'torch',
          'imageio',
          'torch',
          'trimesh',
          'JPype1',
          'progressbar2',
          'graphviz',
          'shapely',
          'Rtree',
          'pyglet<2.0',
          'scikit-image',
          'pytorch-neat @ git+https://github.com/dougsm/PyTorch-NEAT'
          ],
      dependency_links=[
          'https://github.com/dougsm/PyTorch-NEAT/tarball/master#egg=pytorch-neat'
      ],
      packages=['egad'],
      )
