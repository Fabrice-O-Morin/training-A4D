import torch
from pxr import Usd, UsdGeom, Gf, UsdPhysics
import numpy as np

def get_centers_of_mass(blk, device="cuda:0"):
        """
        With articulation, rendered useless by    articulation.data.body_com_pos_w 
        Compute the center of mass of a set of rigid bodies in Isaac Sim.

        Args:
            prim_paths (list[str]): USD paths to rigid body prims.
            local_to (str, optional): prim path to express CoM in its local frame.
                                        If None, returns world coordinates.

        Returns:
            Gf.Vec3d or None: Center of mass position (world or local coordinates).
        """
        COM_list = []
        for prim_list in blk:
            #print(f"\n\nprim_list =\n", prim_list, "\n\n")
            masses = np.array([])
            block_mass = np.array(0.0)
            centers_of_mass = []
            for j in range(3): 
                prim =  prim_list[j]
                
                mass_prim = prim.GetAttribute('physics:mass').Get()# physx_iface.get_rigid_body_mass(prim_path)
                a,b,c = prim.GetAttribute('physics:centerOfMass').Get() # physx_iface.get_rigid_body_center_of_mass(prim_path) # returns tuple of 3 floats
                com_world = [a,b,c]

                # Verification
                #print(f"mass of prim = ", mass_prim)
                #print(type(prim.GetAttribute('physics:centerOfMass').Get()))
                #print(type(a), type(b), type(c))
                #print(f"COM vec3f is \n",prim.GetAttribute('physics:centerOfMass').Get())
                #print(f"com_world is \n", com_world, "\n")

                masses = np.append(masses, mass_prim)
                block_mass += mass_prim
                centers_of_mass.append(com_world)         

                #xform = UsdGeom.Xformable(prim)
                #world_transform = xform.ComputeLocalToWorldTransform(0) # or (Usd.TimeCode.Default())
                #local_transform = world_transform.GetInverse() 

                #com_local_h = Gf.Vec4d(com_world[0], com_world[1], com_world[2], 1.0) * local_transform
                #com_local = Gf.Vec3d(com_local_h[0], com_local_h[1], com_local_h[2])

            mass_prim = np.array(masses)
            centers_of_mass = np.array(centers_of_mass)
            prim_COM = np.average(centers_of_mass, axis=0, weights=mass_prim) #np.dot(mass_prim, centers_of_mass)   
            prim_COM /= block_mass
            COM_list.append(torch.tensor(prim_COM))

            #print("torch.tensor(prim_COM) ", torch.tensor(prim_COM))

        COMs = torch.stack(COM_list, dim=0).to(device)  # shape (n,3)
        return COMs






def print_stage_traverse(stage):
        print(f"\n\n\nSTAGE TRAVERSE\n")
        for prim in stage.Traverse():
            print("Prim:", prim.GetPath(), "Type:", prim.GetTypeName())
        print(f"END STAGE TRAVERSE\n \n \n")



def get_articulation_root(link_prim):
        prim = link_prim
        while prim:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return prim
            prim = prim.GetParent()
        return None

  

def get_all_descendants(prim):
        descendants = np.array()
        for child in prim.GetChildren():
            descendants.append(child)
            descendants.extend(get_all_descendants(child))
        return descendants



def print_hierarchy(prim, indent=0):
        print("  " * indent + prim.GetName())
        for child in prim.GetChildren():
            print_hierarchy(child, indent + 1)


