"""
Script to use in isaacsim to extract a prim from a scene and save it as usda
"""

import omni.usd
from pxr import Usd, UsdUtils, Sdf

# -----------------------------
# Configuration
# -----------------------------
robot_prim_path = "/World/ASSEM4D_with_joints"# prim you want to export
output_usd_path = "/home/fom/ASSEM4Dv2.usda"       # where to save the new USD

# -----------------------------
# Get the current stage
# -----------------------------
stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath(robot_prim_path)
if not robot_prim:
    raise RuntimeError(f"Prim {robot_prim_path} not found in stage")

# -----------------------------
# Create a new stage for the exported prim
# -----------------------------
new_stage = Usd.Stage.CreateNew(output_usd_path)

# Flatten the prim into the new stage
UsdUtils.FlattenLayer(
    stage.GetRootLayer(),            # source layer
    new_stage.GetRootLayer(),        # target layer
    pathToCopy=robot_prim_path       # copy only this prim subtree
)

# Save the new USD
new_stage.GetRootLayer().Save()