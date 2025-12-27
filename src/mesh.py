import os
import cv2
import copy
import torch
import xatlas
import numpy as np
from src.third_party.kaolin import import_mesh, center_points, PBRMaterial, SurfaceMesh

def load_mesh(filename, pbr_material_parameters, use_predefined_texture=False, bsdf_path='assets/bsdf_256_256.bin', device="cuda:0"):
    """Loads a mesh from a file, optionally with predefined textures, and prepares UVs and materials."""
    mesh = import_mesh(filename, with_materials=use_predefined_texture).to(device=device)
    mesh.vertices = 2 * center_points(
        mesh.vertices.unsqueeze(0), normalize=True
    ).squeeze(0)
    
    if use_predefined_texture:
        uvs = mesh.uvs
        indices = mesh.face_uvs_idx
    else:
        vmapping, indices, uvs = xatlas.parametrize(
            mesh.vertices.cpu(),
            mesh.faces.cpu(),
        )
        uvs = torch.tensor(uvs, device=device)
        indices = torch.tensor(indices.astype(np.int64), dtype=torch.long, device=device)

    my_material = PBRMaterial(
        **pbr_material_parameters,
        is_specular_workflow=False
    )
    FG_LUT = torch.from_numpy(
        np.fromfile(bsdf_path, dtype=np.float32).reshape(1, 256, 256, 2)
    ).to(device)
    my_material.FG_LUT = FG_LUT

    return SurfaceMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=uvs,
        face_uvs_idx=indices,
        materials=[my_material,],
        allow_auto_compute=True
    )

def copy_mesh(mesh, texture):
    """Creates a copy of a mesh with a cloned texture and corresponding material LUT."""
    texture_clone = {key: texture[key].detach().clone() for key in texture.keys()}
    my_material = PBRMaterial(
        **texture_clone,
        is_specular_workflow=False
    )
    new_mesh = copy.copy(mesh)
    new_mesh.materials = [my_material, ]
    my_material.FG_LUT = mesh.materials[0].FG_LUT
    return new_mesh

def write_obj(folder, mesh, material):
    """
        Exports a mesh to OBJ/MTL format, writing vertices, normals, texture coordinates, faces,
        and material associations.
    """
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")
        
        v_pos = mesh.vertices.detach().cpu().numpy()
        v_nrm = mesh.vertex_normals.detach().cpu().numpy()
        v_tex = mesh.uvs.detach().cpu().numpy()

        t_pos_idx = mesh.faces.detach().cpu().numpy()
        t_nrm_idx = mesh.faces.detach().cpu().numpy()
        t_tex_idx = mesh.face_uvs_idx.detach().cpu().numpy()

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # Faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    mtl_file = os.path.join(folder, 'mesh.mtl')
    print("Writing material: ", mtl_file)
    save_mtl(mtl_file, material)

    print("Done exporting mesh")

def save_image(fn, x : np.ndarray):
    """Saves an image array to disk, with basic clamping/format handling."""
    try:        
        x = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        if x.shape[-1] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)

@torch.no_grad()
def save_mtl(fn, material):
    """Writes a simple MTL file and associated texture maps to disk."""
    folder = os.path.dirname(fn)     
    with open(fn, "w") as f:
        f.write('newmtl defaultMat\n')                                                               
        if material is not None:
            #f.write('bsdf   %s\n' % material['bsdf'])
            if 'diffuse_texture' in material.keys():
                f.write('map_Kd texture_kd.png\n')                                    
                save_image(
                    os.path.join(folder, 'texture_kd.png'),
                    material['diffuse_texture'].detach().cpu().numpy()
                )
            if 'metallic_texture' in material.keys():
                f.write('map_Pm texture_m.png\n')
                save_image(
                    os.path.join(folder, 'texture_m.png'),
                    material['metallic_texture'].detach().cpu().numpy()
                )
            if 'roughness_texture' in material.keys():
                f.write('map_Pr texture_r.png\n')
                save_image(
                    os.path.join(folder, 'texture_r.png'),
                    material['roughness_texture'].detach().cpu().numpy()
                )
            if 'normals_texture' in material.keys():                                                          
                f.write('bump texture_n.png\n')
                temp_normals = (material['normals_texture'] + 1) * 0.5
                save_image(
                    os.path.join(folder, 'texture_n.png'),
                    temp_normals.detach().cpu().numpy()
                )
        else:              
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')     
            f.write('Tf 1 1 1\n')  
            f.write('Ni 1\n')                                                                        
            f.write('Ns 0\n')
