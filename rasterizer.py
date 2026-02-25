import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class RoomRasterizerCUDA:
    '''Interactive Scene Navigation Engine'''
    def render_room_view(self,camera,gaussians,bg_color=None):
        if bg_color is None:
            bg_color=torch.tensor([0,0,0],dtype=torch.float32,device='cuda')
        settings=GaussianRasterizationSettings(
            image_height=int(camera.H),
            image_width=int(camera.W),
            tanfoxv=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.w2c.cuda(),
            projmatrix=camera.full_proj.cuda(),
            sh_degree=3,
            campos=camera.pos.cuda(),
            prefiltered=True,
            debug=False
        )
        rasterizer=GaussianRasterizationSettings(raster_settings=settings)
        rendered_image,radii=rasterizer(
            means3D=gaussians.xyz,
            means2D=torch.zeros_like(gaussians.xyz,device='cuda'),
            shs=gaussians.shs,
            colors_precomp=None,
            opacities=gaussians.opacity,
            scales=gaussians.scales,
            rotations=gaussians.rotations
        )
        return {"render":rendered_image,"viewspace_points":radii}
    