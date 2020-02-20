import sys
import os
import tempfile
import shutil
from pathlib2 import Path

from autolab_core import YamlConfig
import dexnet
import dexnet.database.mesh_processor as mp
from dexnet.grasping.gripper import RobotGripper
from dexnet.grasping.grasp_sampler import AntipodalGraspSampler
from dexnet.grasping.graspable_object import GraspableObject3D
from dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from dexnet.grasping.grasp_quality_function import GraspQualityFunctionFactory


mesh_file = sys.argv[-1]

# Use local config file
config_path = (Path(__file__).parent/'cfg/dexnet_api_settings.yaml').resolve()
config = YamlConfig(str(config_path))

# Requires data from the dexnet project.
os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))

# Cache directory
mp_cache = tempfile.mkdtemp()
mesh_processor = mp.MeshProcessor(mesh_file, mp_cache)
mesh_processor.generate_graspable(config)
shutil.rmtree(mp_cache)

gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=config['gripper_dir'])
sampler = AntipodalGraspSampler(gripper, config)
obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)

grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])

metric_config = GraspQualityConfigFactory.create_config(config['metrics']['ferrari_canny'])
quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, metric_config)

qualities = [quality_fn(g).quality for g in grasps]

qualities.sort()

# 75% percentile
q75 = qualities[int(len(qualities)*0.75)]

# Scale 0-100
scale_min = 0.0005
scale_max = 0.004
q75 = max(min(q75, scale_max), scale_min)
q75 = int((q75 - scale_min)/(scale_max - scale_min) * 100)

exit(q75+2)  # add 2 to differentiate from 0 and 1, which are legit error return codes.
