from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
link_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Wheels_and_axleAssem4d1_mY0Vd0"
link_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Drone_body_with_flangesAssem4d1_ik0xp0"
link_prim = stage.GetPrimAtPath(link_path)

for child in link_prim.GetChildren():
    print(child.GetPath(), child.GetTypeName(), child.GetAppliedSchemas())
