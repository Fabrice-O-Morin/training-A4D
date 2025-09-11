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



# -----------------------------
# 1️⃣ Start or get IsaacSim app
# -----------------------------
sim_app = omni.isaac.kit.SimulationApp({"headless": False})

# -----------------------------
# 2️⃣ Get the stage
# -----------------------------
stage = omni.usd.get_context().get_stage()

# -----------------------------
# 3️⃣ Function to collect rigid bodies
# -----------------------------
def get_rigid_bodies(container_prim_path="/Root"):
    container = stage.GetPrimAtPath(container_prim_path)
    if not container.IsValid():
        print("Container prim not found:", container_prim_path)
        return []

    rigid_bodies = []
    for prim in container.GetAllChildren():
        if prim.HasAPI("PhysxRigidBodyAPI"):
            rigid_bodies.append(prim)
        else:
            # Recursively check children
            rigid_bodies.extend(get_rigid_bodies(prim.GetPath()))
    return rigid_bodies

# -----------------------------
# 4️⃣ Step simulation to initialize physics
# -----------------------------
# Run a few frames so PhysX computes COM
for _ in range(5):
    sim_app.update()

# -----------------------------
# 5️⃣ Query COM for each rigid body
# -----------------------------
rigid_bodies = get_rigid_bodies("/Root")
for rb in rigid_bodies:
    com_attr = rb.GetAttribute("physics:centerOfMass")
    mass_attr = UsdPhysics.RigidBodyAPI(rb).GetMassAttr()
    mass = mass_attr.Get() if mass_attr else None
    com = com_attr.Get() if com_attr else None
    print(f"RigidBody: {rb.GetName()}, Mass: {mass}, COM: {com}")

# -----------------------------
# 6️⃣ Keep simulation running if needed
# -----------------------------
# sim_app.close()  # call this if you want to close the simulation

