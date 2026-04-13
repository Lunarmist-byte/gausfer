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
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.w2c.cuda().mT,
            projmatrix=camera.full_proj.cuda().mT,
            sh_degree=gaussians.active_sh_degree,
            campos=camera.pos.cuda(),
            prefiltered=False,
            debug=False
        )
        # Create viewspace points tensor to capture gradients for densification
        means2D = torch.zeros_like(gaussians.xyz, dtype=torch.float32, device='cuda', requires_grad=True)
        try:
            means2D.retain_grad()
        except:
            pass

        # Pre-flight NaN/Inf check: catches exploded Gaussians before they kill the CUDA kernel
        xyz = gaussians.xyz
        scales = gaussians.scales
        rotations = gaussians.rotations
        opacities = gaussians.opacity
        shs = gaussians.shs

        # Comprehensive FINITE checks
        if not torch.isfinite(xyz).all():
            raise RuntimeError(f"NaN/Inf in XYZ positions")
        if not torch.isfinite(scales).all():
            raise RuntimeError(f"NaN/Inf in scales")
        if not torch.isfinite(rotations).all():
            raise RuntimeError(f"NaN/Inf in rotations")
        if not torch.isfinite(opacities).all():
            raise RuntimeError(f"NaN/Inf in opacities")
        if not torch.isfinite(shs).all():
            raise RuntimeError(f"NaN/Inf in SH coefficients")
            
        # Hard Scale check: extremely large splats kill the tile rasterizer
        if (scales > 5.0).any():
            raise RuntimeError(f"Explosive Gaussian scale detected (>5.0 world units)")

        rasterizer=GaussianRasterizer(raster_settings=settings)
        rendered_image,radii=rasterizer(
            means3D=xyz,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations
        )
        return {"render":rendered_image, "viewspace_points": means2D, "radii": radii}