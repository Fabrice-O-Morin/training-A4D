from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()

# Path to the link that is failing to wrap
link_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Wheels_and_axleAssem4d1_mY0Vd0"
link_path = "/Root/ASSEM4D_with_joints/ASSEM4D/tn__Drone_body_with_flangesAssem4d1_ik0xp0"
link_prim = stage.GetPrimAtPath(link_path)

if not link_prim.IsValid():
    print(f"Invalid prim path: {link_path}")
else:
    print(f"Checking children of {link_path}:")

    for child in link_prim.GetChildren():
        type_name = child.GetTypeName()
        schemas = child.GetAppliedSchemas()

        # Criteria for RigidObject eligibility
        is_mesh_or_xform = type_name in ["Mesh", "Xform"]
        has_physics_rigid = "PhysicsRigidBodyAPI" in schemas or "PhysicsArticulationRootAPI" in schemas
        is_container = type_name in ["Scope", "Xform"] and len(child.GetChildren()) > 0

        eligible = is_mesh_or_xform and not has_physics_rigid

        status = "✅ Eligible" if eligible else "❌ NOT eligible"
        print(f" - {child.GetPath()} | type: {type_name}, schemas: {schemas} → {status}")

