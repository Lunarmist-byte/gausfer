import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
def render_room_view(viewpoint_camera,pc,pipe,bg_color):
    '''
    Interactive Scene Navigation
    '''
    settings=