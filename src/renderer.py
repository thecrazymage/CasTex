import torch
import nvdiffrast
import nvdiffrast.torch as dr

class Renderer:
    def __init__(self):
        """Initializes the CUDA rasterization context for rendering."""
        self.glctx = dr.RasterizeCudaContext()
    
    def _interpolate_attributes(self, rast_out, rast_out_db, mesh):
        """
            Interpolates vertex attributes like normals, tangents, and UV coordinates 
            across the rasterized face fragments.
        """
        normals = nvdiffrast.torch.interpolate(
            mesh.get_or_compute_attribute('vertex_normals', should_cache=False),
            rast_out,
            mesh.faces.int(),
            rast_db=rast_out_db,
            diff_attrs='all',
        )[0]
        tangents = nvdiffrast.torch.interpolate(
            mesh.get_or_compute_attribute('vertex_tangents', should_cache=False),
            rast_out,
            mesh.faces.int(),
            rast_db=rast_out_db,
            diff_attrs='all',
        )[0]
        bitangents = torch.nn.functional.normalize(torch.cross(tangents, normals, dim=-1), dim=-1)
        # get uvs
        texc, texd = nvdiffrast.torch.interpolate(
            mesh.uvs,
            rast_out,
            mesh.face_uvs_idx.int(),
            rast_db=rast_out_db,
            diff_attrs='all',
        )
        return normals, tangents, bitangents, texc, texd

    def _get_ray_dirs(self, camera, device='cuda:0'):
        """Computes the direction vectors for rays passing through each pixel of the camera."""
        num_cameras = len(camera)
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(camera.height, device=device),
            torch.arange(camera.width, device=device),
        )
        pixel_x = pixel_x + 0.5
        pixel_x = pixel_x.unsqueeze(0) - camera.x0.view(-1, 1, 1)
        pixel_x = 2 * (pixel_x / camera.width) - 1.0

        pixel_y = pixel_y + 0.5
        pixel_y = pixel_y.unsqueeze(0) - camera.y0.view(-1, 1, 1)
        pixel_y = 2 * (pixel_y / camera.height) - 1.0

        ray_dir = torch.stack((pixel_x * camera.tan_half_fov.view(-1, 1, 1),
                            -pixel_y * camera.tan_half_fov.view(-1, 1, 1),
                            -torch.ones_like(pixel_x)), dim=-1)
        ray_dir = ray_dir.reshape(num_cameras, -1, 3)  # Flatten grid rays to 1D array
        ray_orig = torch.zeros_like(ray_dir)
        # Transform from camera to world coordinates
        ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
        ray_dir = ray_dir.reshape(-1, camera.height, camera.width, 3)
        return ray_dir

    def render(self, mesh, camera, light, random_background=True, val_background=False, device='cuda:0'):
        """
            Executes the full rendering pipeline, including vertex transformation, rasterization, 
            attribute interpolation, texture mapping, and PBR shading to produce the final image.
        """
        # transform mesh
        vertices_camera = camera.extrinsics.transform(mesh.vertices)
        vertices_clip = camera.intrinsics.project(vertices_camera)
        faces_int = mesh.faces.int()

        rast_out, rast_out_db = nvdiffrast.torch.rasterize(
            self.glctx,
            vertices_clip,
            faces_int,
            (camera.height, camera.width),
        )
        rast_out = torch.flip(rast_out, dims=(1,))
        rast_out_db = torch.flip(rast_out_db, dims=(1,))
        mask = torch.clamp(rast_out[..., -1:], 0, 1)
        # interpolate normals, tangents & bitangents
        normals, tangents, bitangents, texc, texd = self._interpolate_attributes(
            rast_out,
            rast_out_db,
            mesh,
        )
        # texturing
        material = mesh.materials[0]

        def _proc_channel(texture_image):
            """
                Samples a specific texture map using linear mipmap filtering 
                based on the interpolated UV coordinates.
            """
            if texture_image is None:
                return None
            return nvdiffrast.torch.texture(
                texture_image[None, ...],
                texc,
                texd,
                filter_mode='linear-mipmap-linear',
                max_mip_level=9
            )
        mapped_albedo = _proc_channel(material.diffuse_texture)
        mapped_normal = _proc_channel(material.normals_texture)
        mapped_metallic = _proc_channel(material.metallic_texture)
        mapped_roughness = _proc_channel(material.roughness_texture)
        # shading
        if mapped_normal is not None:
            shading_normals = torch.nn.functional.normalize(
                tangents * mapped_normal[..., :1]
                - bitangents * mapped_normal[..., 1:2]
                + normals * mapped_normal[..., 2:3],
                dim=-1,
            )
        else:
            shading_normals = normals
        diffuse_light = light(shading_normals)

        if mapped_metallic is not None and mapped_roughness is not None:
            viewdirs = -self._get_ray_dirs(camera, device=device)
            n_dot_v = (shading_normals * viewdirs).sum(-1, keepdim=True)
            reflective = n_dot_v * shading_normals * 2 - viewdirs

            roughness = torch.clamp(mapped_roughness, min=1e-3)
            specular_light = light(reflective, roughness)
        
            diffuse_albedo = (1 - mapped_metallic) * mapped_albedo
            fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
            fg = dr.texture(
                mesh.materials[0].FG_LUT,
                fg_uv.reshape(1, -1, 1, 2).contiguous(),
                filter_mode='linear',
                boundary_mode='clamp',
                ).reshape(*roughness.shape[:-1], 2)
            F0 = (1. - mapped_metallic) * 0.04 + mapped_metallic * mapped_albedo
            specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]
            output_image = diffuse_light * diffuse_albedo + specular_light * specular_albedo
        else:
            diffuse_albedo = mapped_albedo
            output_image = diffuse_light * diffuse_albedo

        if random_background:
            background_color = torch.rand(3, device=device)
        else:
            background_color = torch.tensor([0.05, 0.05, 0.05], device=device)
        
        if val_background:
            background_color = torch.tensor([0.99, 0.99, 0.99], device=device) 

        output_image = torch.where(mask == 1, output_image, background_color)
        return output_image