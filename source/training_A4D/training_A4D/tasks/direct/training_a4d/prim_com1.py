"""
Script to use in isaacsim to extract a prim from a scene and save it as usda
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdUtils, Sdf
from omni.isaac.core import SimulationContext

# -----------------------------
# Configuration
# -----------------------------
prim_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Drone_body_with_flangesAssem4d1_ik0xp0" # prim 
prim_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Drone_body_with_flangesAssem4d1_ik0xp0/tn__MODELSimpleDrone11_sQI"

#output_usd_path = "/home/fom/ASSEM4Dv2.usda"       # where to save the new USD


# 1️⃣ Get the stage
stage = omni.usd.get_context().get_stage()


# 2️⃣ Get your prim by path
prim = stage.GetPrimAtPath(prim_path)


# 3️⃣ Check if the prim exists
if not prim.IsValid():
    print("Prim not found at path:", prim_path)
else:
    app = omni.kit.app.get_app()
    if app:
        app.update()  # this steps the simulation and physics once
    else:
        print("Cannot get Kit app")


# 4️⃣ Get the physics:centerOfMass attribute
com_attr = prim.GetAttribute('physics:centerOfMass')
if com_attr:
    com = com_attr.Get()
    print("Center of Mass:", com)
else:
    print("Prim has no physics:centerOfMass attribute")
