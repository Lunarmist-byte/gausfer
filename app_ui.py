import sys
import os
import cv2
import tempfile
import torch
import traceback
from datetime import datetime

import torch.optim as optim
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QProgressBar, QSlider, QSpinBox, QFileDialog, 
                             QPlainTextEdit, QMessageBox, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from pose_estimator import PoseEstimator
from dataset_loader import RoomDatasetLoader
from nerf_model import RoomNeRF
from nerf_trainer import NeRFTrainer
from bridge_converter import NeRFToGaussianBridge
from HybridCoOptimizer import HybridCoOptimizer
from rasterizer import RoomRasterizerCUDA
from gaussian_model import GaussianModel
from main import ViewCamera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class OrbitCamera:
    def __init__(self, distance=5.0, target=None):
        self.distance = distance
        self.target = target if target is not None else torch.tensor([0.0, 0.0, 0.0], device='cuda')
        self.yaw = 0.0
        self.pitch = 0.0
        
    def get_c2w(self):
        # Calculate camera position based on orbit
        cos_p = np.cos(self.pitch)
        sin_p = np.sin(self.pitch)
        cos_y = np.cos(self.yaw)
        sin_y = np.sin(self.yaw)
        
        offset = torch.tensor([
            self.distance * cos_p * sin_y,
            self.distance * sin_p,
            self.distance * cos_p * cos_y
        ], device='cuda', dtype=torch.float32)
        
        pos = self.target + offset
        
        # Build look-at matrix
        forward = -offset / torch.norm(offset)
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
        right = torch.cross(up, forward, dim=0)
        right = right / torch.norm(right)
        up = torch.cross(forward, right, dim=0)
        
        c2w = torch.eye(4, device='cuda')
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = pos
        return c2w

class GaussianVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.gaussians = None
        self.rasterizer = RoomRasterizerCUDA()
        self.camera = OrbitCamera()
        self.scale_modifier = 1.0
        self.last_mouse_pos = QPoint()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_render)
        self.current_image = None
        self.needs_render = False

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def load_model(self, path):
        if not os.path.exists(path):
            return False
        if self.gaussians is None:
            self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_checkpoint(path)
        self.needs_render = True
        self.update()
        return True

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            diff = event.pos() - self.last_mouse_pos
            self.camera.yaw -= diff.x() * 0.01
            self.camera.pitch = np.clip(self.camera.pitch + diff.y() * 0.01, -np.pi/2 + 0.1, np.pi/2 - 0.1)
            self.last_mouse_pos = event.pos()
            self.needs_render = True
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            diff = event.pos() - self.last_mouse_pos
            # Simple pan logic
            self.camera.target[1] += diff.y() * 0.01
            self.last_mouse_pos = event.pos()
            self.needs_render = True
            self.update()

    def wheelEvent(self, event):
        self.camera.distance = max(0.1, self.camera.distance - event.angleDelta().y() * 0.005)
        self.needs_render = True
        self.update()

    def update_render(self):
        if not self.needs_render or self.gaussians is None:
            return
            
        # Create a temporary camera object for the rasterizer
        w, h = self.width(), self.height()
        # Mock focal length for 60 deg horizontal FOV
        fx = w / (2.0 * np.tan(np.deg2rad(30)))
        K = torch.tensor([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], device='cuda')
        c2w = self.camera.get_c2w()
        
        view_cam = ViewCamera(h, w, K, c2w)
        
        # Manually inject scale modifier into rasterizer settings
        settings = GaussianRasterizationSettings(
            image_height=int(view_cam.H),
            image_width=int(view_cam.W),
            tanfovx=view_cam.tanfovx,
            tanfovy=view_cam.tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda'),
            scale_modifier=self.scale_modifier,
            viewmatrix=view_cam.w2c.cuda().mT,
            projmatrix=view_cam.full_proj.cuda().mT,
            sh_degree=self.gaussians.active_sh_degree,
            campos=view_cam.pos.cuda(),
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        render_data, _ = rasterizer(
            means3D=self.gaussians.xyz,
            means2D=torch.zeros_like(self.gaussians.xyz, device='cuda'),
            shs=self.gaussians.shs,
            colors_precomp=None,
            opacities=self.gaussians.opacity,
            scales=self.gaussians.scales,
            rotations=self.gaussians.rotations
        )
        
        img_tensor = render_data.detach().clamp(0, 1)
        
        # Convert torch (3, H, W) to numpy (H, W, 3)
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        self.current_image = QImage(img_np.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self.needs_render = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.current_image:
            painter.drawImage(0, 0, self.current_image)
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No model loaded or rendering...")
        
    def start(self):
        self.timer.start(33) # ~30 FPS

    def stop(self):
        self.timer.stop()

# Config
SAVE_DIR = "./images"
os.makedirs(SAVE_DIR, exist_ok=True)

class EmittingStream(QObject):
    """Custom stream to redirect stdout to PyQT TextEdit"""
    textWritten = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def write(self, text):
        self.textWritten.emit(str(text))
        
    def flush(self):
        pass

class VideoExtractorThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_extraction = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, video_path, fps_target, blur_strictness):
        super().__init__()
        self.video_path = video_path
        self.fps_target = fps_target
        self.blur_strictness = blur_strictness

    def is_blurry(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def run(self):
        try:
            self.log.emit(f"Starting extraction for {self.video_path}...")
            cap = cv2.VideoCapture(self.video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                self.error.emit("Failed to read video framerate.")
                return

            frame_interval = max(1, int(video_fps / self.fps_target))
            count = 0
            saved = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_interval == 0:
                    if not self.is_blurry(frame, self.blur_strictness):
                        filepath = os.path.join(SAVE_DIR, f"auto_frame_{saved:04d}.jpg")
                        cv2.imwrite(filepath, frame)
                        saved += 1
                count += 1
                # Update progress bar (0-100)
                progress_val = int((count / total_frames) * 100)
                self.progress.emit(progress_val)
            
            cap.release()
            self.log.emit(f"Successfully extracted {saved} sharp frames.")
            self.finished_extraction.emit(saved)
        except Exception as e:
            self.error.emit(f"Video Extraction Error: {str(e)}")

import time

class PipelineTrainingThread(QThread):
    progress_colmap = pyqtSignal(int)
    progress_nerf = pyqtSignal(int)
    progress_3dgs = pyqtSignal(int)
    log = pyqtSignal(str)
    eta = pyqtSignal(str)
    finished_training = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_dir='./images', num_epochs=10, num_steps=5000, resume_checkpoint=False):
        super().__init__()
        self.image_dir = image_dir
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.resume_checkpoint = resume_checkpoint
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            image_dir = self.image_dir

            output_dir = './output'
            colmap_out = os.path.join(output_dir, 'sparse')

            # 1. Phase: COLMAP
            self.progress_colmap.emit(10)
            colmap_model_path = os.path.join(colmap_out, "0")
            if not os.path.exists(os.path.join(colmap_model_path, "cameras.txt")):
                self.log.emit("Fresh images detected. Running COLMAP...")
                estimator = PoseEstimator(image_path=image_dir, output_path=output_dir)
                estimator.run_colmap()
            else:
                self.log.emit("Camera poses already found. Skipping COLMAP.")
            self.progress_colmap.emit(100)

            ckpt_nerf_path = os.path.join(output_dir, "nerf_checkpoint.pth")
            ckpt_gauss_path = os.path.join(output_dir, "gauss_checkpoint.pth")
            
            self.log.emit("Loading Dataset...")
            dataset = RoomDatasetLoader(colmap_dir=colmap_model_path, images_dir=image_dir)
            nerf = RoomNeRF().cuda()
            gaussians = GaussianModel(sh_degree=3)

            if self.resume_checkpoint and os.path.exists(ckpt_nerf_path) and os.path.exists(ckpt_gauss_path):
                self.log.emit("Found Checkpoint Data! Resuming directly to Phase 3 (Co-Optimization)...")
                self.progress_nerf.emit(100)
                self.progress_3dgs.emit(5)
                
                nerf.load_state_dict(torch.load(ckpt_nerf_path))
                gaussians.load_checkpoint(ckpt_gauss_path)
            else:
                if self.resume_checkpoint:
                    self.log.emit("Resume requested, but no checkpoint found. Starting from scratch...")
                    
                # 2. Phase: NeRF Warmup
                self.progress_nerf.emit(5)
                self.log.emit("Initializing NeRF...")
                trainer = NeRFTrainer(nerf)
                
                num_epochs = self.num_epochs
                self.log.emit(f"Starting NeRF Warmup ({num_epochs} Epochs)...")
                start_time = time.time()
                for epoch in range(num_epochs):
                    if not self.is_running:
                        self.log.emit("Training stopped by user during NeRF Warmup.")
                        self.eta.emit("ETA: Stopped")
                        self.finished_training.emit()
                        return
                    for i in range(len(dataset.images)):
                        if not self.is_running:
                            break
                        img, poses = dataset.get_training_batch(i)
                        loss = trainer.train_step(img, poses)
                        
                        # Update ETA continuously per image
                        if i > 0:
                            elapsed = time.time() - start_time
                            total_iters_done = epoch * len(dataset.images) + i
                            avg_time_per_iter = elapsed / total_iters_done
                            iters_remaining = num_epochs * len(dataset.images) - total_iters_done
                            rem = avg_time_per_iter * iters_remaining
                            m, s = divmod(int(rem), 60)
                            self.eta.emit(f"ETA (NeRF): {m}m {s}s")
                            
                    self.log.emit(f"NeRF Epoch {epoch} Loss: {loss:.4f}")
                    self.progress_nerf.emit(int(((epoch + 1) / num_epochs) * 100))
                    trainer.step_scheduler() # Decay learning rate
                
                # 3. Phase: Bridging and 3DGS
                self.progress_3dgs.emit(5)
                self.log.emit("Seeding Gaussians via Neural Density...")
                self.eta.emit("ETA: Bridging Arrays (Please Wait)...")
                bbox = [[-5, -5, -5], [5, 5, 5]]
                bridge = NeRFToGaussianBridge(nerf, bbox)
                init_xyz, init_rgb = bridge.generate_initial_gaussians()
                
                gaussians.initialize_from_nerf(init_xyz, init_rgb)
                
                # Save checkpoints for future resumes
                torch.save(nerf.state_dict(), ckpt_nerf_path)
                gaussians.save_checkpoint(ckpt_gauss_path)
            
            # Calculate scene radius for LR scaling (standard 3DGS practice)
            with torch.no_grad():
                center = gaussians.xyz.mean(dim=0)
                spatial_lr_scale = torch.norm(gaussians.xyz - center, dim=-1).max().item()
                spatial_lr_scale = max(1.0, min(spatial_lr_scale, 20.0)) # Guard rails
            self.log.emit(f"  Automatic Spatial Scale: {spatial_lr_scale:.2f}")

            # Per-parameter learning rates (critical for 3DGS quality)
            # Position LR is scaled by the scene radius
            param_groups = [
                {"params": [gaussians._xyz], "lr": 0.00016 * spatial_lr_scale, "name": "xyz"},
                {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
                {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
                {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
                {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
                {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
            ]
            gaussians_optim = optim.Adam(param_groups, lr=0.0, eps=1e-15)
            self.log.emit("Starting Hybrid Co-Optimization...")
            co_optimizer = HybridCoOptimizer(nerf, gaussians)
            rasterizer = RoomRasterizerCUDA()

            num_steps = self.num_steps
            start_time = time.time()
            consecutive_failures = 0
            max_consecutive_failures = 5
            for step in range(num_steps):
                if not self.is_running:
                    self.log.emit("Training stopped by user during 3DGS Optimization.")
                    self.eta.emit("ETA: Stopped")
                    # Auto-save on user stop so progress is never lost
                    self.log.emit("Auto-saving checkpoint on stop...")
                    torch.save(nerf.state_dict(), ckpt_nerf_path)
                    gaussians.save_checkpoint(ckpt_gauss_path)
                    break
                
                try:
                    # 0. Safety Guard: Enforce physical constraints BEFORE any CUDA call
                    # This prevents the rasterizer from even seeing 'explosive' parameters
                    gaussians.constrain_parameters()
                    
                    # 1. Exponential LR decay for positions
                    progress = step / num_steps
                    current_xyz_lr = (0.00016 * spatial_lr_scale) * ((0.01) ** progress)
                    for param_group in gaussians_optim.param_groups:
                        if param_group["name"] == "xyz":
                            param_group["lr"] = current_xyz_lr

                    gaussians_optim.zero_grad()
                    idx = step % len(dataset.images)
                    ground_truth_image, (H, W, K, c2w) = dataset.get_training_batch(idx)

                    view_cam = ViewCamera(H, W, K, c2w)
                    old_count = gaussians.xyz.shape[0]
                    loss, _ = co_optimizer.step(view_cam=view_cam, ground_truth_image=ground_truth_image, render_func=rasterizer.render_room_view, step=step)
                    
                    if gaussians.xyz.shape[0] != old_count:
                        # Rebuild optimizer with per-parameter LRs after point count change
                        from main import update_optimizer
                        param_groups = [
                            {"params": [gaussians._xyz], "lr": current_xyz_lr, "name": "xyz"},
                            {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
                            {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
                            {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
                            {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
                            {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
                        ]
                        gaussians_optim = update_optimizer(gaussians_optim, param_groups)
                        torch.cuda.empty_cache() # Stabilize memory after growth

                    gaussians_optim.step()
                    consecutive_failures = 0  
                except (RuntimeError, Exception) as step_err:
                    torch.cuda.empty_cache()
                    err_msg = str(step_err)
                    
                    # --- Comprehensive Rescue ---
                    self.log.emit(f"  [RESCUE] Step {step} triggered error: {err_msg[:60]}")
                    
                    with torch.no_grad():
                        # Force constraints to scrub NaNs/Infs that caused the crash
                        gaussians.constrain_parameters()
                        
                        # Check if point count is still the same for optimizer safety
                        from main import update_optimizer
                        param_groups = [
                            {"params": [gaussians._xyz], "lr": current_xyz_lr, "name": "xyz"},
                            {"params": [gaussians._features_dc], "lr": 0.0025, "name": "f_dc"},
                            {"params": [gaussians._features_rest], "lr": 0.000125, "name": "f_rest"},
                            {"params": [gaussians._opacity], "lr": 0.05, "name": "opacity"},
                            {"params": [gaussians._scaling], "lr": 0.005, "name": "scaling"},
                            {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
                        ]
                        gaussians_optim = update_optimizer(gaussians_optim, param_groups)
                        
                        # Reset co-optimization buffers to safe state
                        co_optimizer.xyz_gradient_accum = torch.zeros((gaussians.xyz.shape[0], 1), device='cuda')
                        co_optimizer.denom = torch.zeros((gaussians.xyz.shape[0], 1), device='cuda')
                        co_optimizer.max_radii2D = torch.zeros((gaussians.xyz.shape[0]), device='cuda')
                    
                    consecutive_failures += 1
                    
                    # Log to dedicated error file for deep diagnostics
                    try:
                        with open("error_report.log", "a") as f:
                            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP {step} FAILURE\n")
                            f.write(f"Error: {err_msg}\n")
                            f.write(traceback.format_exc())
                            f.write("-" * 40 + "\n")
                    except:
                        pass

                    if consecutive_failures >= max_consecutive_failures:
                        self.log.emit(f"  [FATAL] Unrecoverable failure at step {step}. Check error_report.log for details.")
                        torch.save(nerf.state_dict(), ckpt_nerf_path)
                        gaussians.save_checkpoint(ckpt_gauss_path)
                        self.error.emit(f"Critical training failure at step {step}. State saved.")
                        self.finished_training.emit()
                        return
                    continue
                
                if step % 10 == 0:
                    image_name = dataset.images[idx]['name']
                    self.log.emit(f"Step {step}/{num_steps} | Loss: {loss:.4f} | View {idx} ({image_name}) | Splats: {gaussians.xyz.shape[0]} | SH: {gaussians.active_sh_degree}")
                
                # Auto-checkpoint every 500 steps to prevent progress loss
                if step > 0 and step % 500 == 0:
                    self.log.emit(f"  [Checkpoint] Auto-saving at step {step}...")
                    torch.save(nerf.state_dict(), ckpt_nerf_path)
                    gaussians.save_checkpoint(ckpt_gauss_path)
                
                # Update 3DGS Progress
                if step % 50 == 0:
                    self.progress_3dgs.emit(int((step / num_steps) * 100))
                
                # Update ETA continuously
                if step > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / step
                    rem = avg_time * (num_steps - step)
                    m, s = divmod(int(rem), 60)
                    self.eta.emit(f"ETA (3DGS): {m}m {s}s")
            
            self.progress_3dgs.emit(100)
            self.eta.emit("ETA: Done")
            
            # Final save
            torch.save(nerf.state_dict(), ckpt_nerf_path)
            gaussians.save_checkpoint(ckpt_gauss_path)
            
            out_ply = os.path.join(output_dir, "point_cloud.ply")
            gaussians.save_ply(out_ply)
            
            self.log.emit(f"\n Full Room Reconstruction Done! Saved 3D point cloud to: {out_ply}")
            
            try:
                import subprocess
                if sys.platform == "win32":
                    os.startfile(os.path.abspath(out_ply))
                elif sys.platform == "darwin":
                    subprocess.call(["open", os.path.abspath(out_ply)])
                else:
                    subprocess.call(["xdg-open", os.path.abspath(out_ply)])
            except Exception as e:
                self.log.emit(f"Could not automatically open PLY file: {str(e)}")
                
            self.finished_training.emit()

        except Exception as e:
            # Emergency checkpoint on ANY crash
            try:
                self.log.emit("[CRASH] Saving emergency checkpoint before exit...")
                torch.save(nerf.state_dict(), ckpt_nerf_path)
                gaussians.save_checkpoint(ckpt_gauss_path)
                self.log.emit("[CRASH] Emergency checkpoint saved. Use 'Resume from Checkpoint' to continue.")
            except Exception:
                pass
            self.error.emit(f"Pipeline Error: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gausfer")
        self.setGeometry(100, 100, 900, 700)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tab_extraction = QWidget()
        self.tab_training = QWidget()
        self.tab_viewer = QWidget()
        self.tabs.addTab(self.tab_extraction, "1. Data Input")
        self.tabs.addTab(self.tab_training, "2. Execution Pipeline")
        self.tabs.addTab(self.tab_viewer, "3. Real-Time View")
        main_layout.addWidget(self.tabs)
        
        self.custom_image_dir = './images'

        self.setup_extraction_tab()
        self.setup_training_tab()
        self.setup_visualizer_tab()
        
        # Console Group
        console_group = QGroupBox("Live Output Log")
        console_layout = QVBoxLayout()
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        main_layout.addWidget(console_group, stretch=1)

        # Redirect Stdout
        self.redirect_stdout()

    def setup_extraction_tab(self):
        layout = QVBoxLayout(self.tab_extraction)
        
        # Option 1: Video Group
        video_group = QGroupBox("Option 1: Video Extraction")
        video_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        video_layout = QVBoxLayout()
        
        lbl_desc = QLabel("Upload a room walkthrough video. The system will automatically extract sharp frames for the dataset.")
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("font-weight: normal; font-size: 12px;")
        video_layout.addWidget(lbl_desc)
        
        # Video Selection
        file_layout = QHBoxLayout()
        self.lbl_file = QLabel("No Video Selected")
        self.lbl_file.setStyleSheet("font-weight: normal; font-size: 12px;")
        btn_select = QPushButton("Select Video")
        btn_select.setStyleSheet("font-weight: normal; font-size: 12px;")
        btn_select.clicked.connect(self.select_video)
        file_layout.addWidget(self.lbl_file)
        file_layout.addWidget(btn_select)
        video_layout.addLayout(file_layout)
        
        # Settings
        settings_layout = QHBoxLayout()
        # FPS
        fps_layout = QVBoxLayout()
        lbl_fps = QLabel("Target FPS:")
        lbl_fps.setStyleSheet("font-weight: normal; font-size: 12px;")
        fps_layout.addWidget(lbl_fps)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 10)
        self.spin_fps.setValue(2)
        fps_layout.addWidget(self.spin_fps)
        settings_layout.addLayout(fps_layout)
        
        # Blur Strictness
        blur_layout = QVBoxLayout()
        blur_lbl = QLabel("Blur Strictness Threshold:")
        blur_lbl.setStyleSheet("font-weight: normal; font-size: 12px;")
        blur_layout.addWidget(blur_lbl)
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(50, 300)
        self.slider_blur.setValue(100)
        blur_layout.addWidget(self.slider_blur)
        settings_layout.addLayout(blur_layout)
        
        video_layout.addLayout(settings_layout)
        
        # Extraction control
        self.btn_extract = QPushButton("Start Extraction")
        self.btn_extract.setMinimumHeight(40)
        self.btn_extract.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.btn_extract.clicked.connect(self.start_extraction)
        video_layout.addWidget(self.btn_extract)
        
        self.prog_extraction = QProgressBar()
        video_layout.addWidget(self.prog_extraction)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        layout.addSpacing(10)
        
        # Option 2: Image Folder Group
        folder_group = QGroupBox("Option 2: Use Existing Images Folder")
        folder_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        folder_layout = QVBoxLayout()
        
        lbl_desc2 = QLabel("Already have extracted frames? Select the directory containing your images (bypasses video extraction).")
        lbl_desc2.setWordWrap(True)
        lbl_desc2.setStyleSheet("font-weight: normal; font-size: 12px;")
        folder_layout.addWidget(lbl_desc2)
        
        f_layout = QHBoxLayout()
        self.lbl_folder = QLabel("Current Folder: ./images")
        self.lbl_folder.setStyleSheet("font-weight: normal; font-size: 12px;")
        btn_select_folder = QPushButton("Browse Folder")
        btn_select_folder.setStyleSheet("font-weight: normal; font-size: 12px;")
        btn_select_folder.clicked.connect(self.select_folder)
        f_layout.addWidget(self.lbl_folder)
        f_layout.addWidget(btn_select_folder)
        folder_layout.addLayout(f_layout)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        layout.addStretch()

        self.video_path = None

    def setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)
        
        lbl_desc = QLabel("Execute the Hybrid Gausfer NeRF+3DGS Pipeline.")
        layout.addWidget(lbl_desc)
        
        # Settings
        params_layout = QHBoxLayout()
        lbl_epochs = QLabel("NeRF Epochs:")
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(30)
        params_layout.addWidget(lbl_epochs)
        params_layout.addWidget(self.spin_epochs)
        
        lbl_steps = QLabel("Optimization Steps:")
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(100, 50000)
        self.spin_steps.setSingleStep(100)
        self.spin_steps.setValue(15000)
        params_layout.addWidget(lbl_steps)
        params_layout.addWidget(self.spin_steps)
        
        layout.addLayout(params_layout)
        
        self.chk_resume = QCheckBox("Resume from Checkpoint (Skip Phase 1 & Phase 2 if available)")
        layout.addWidget(self.chk_resume)
        
        # Status Label
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        self.lbl_eta = QLabel("ETA: N/A")
        self.lbl_eta.setStyleSheet("font-weight: bold;")
        self.lbl_eta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_eta)
        layout.addLayout(status_layout)
        
        # COLMAP Progress
        layout.addWidget(QLabel("Phase 1: COLMAP Pose Estimation"))
        self.prog_colmap = QProgressBar()
        layout.addWidget(self.prog_colmap)
        
        # NeRF Progress
        layout.addWidget(QLabel("Phase 2: NeRF Warmup"))
        self.prog_nerf = QProgressBar()
        layout.addWidget(self.prog_nerf)
        
        # 3DGS Progress
        layout.addWidget(QLabel("Phase 3: Hybrid Co-Optimization"))
        self.prog_3dgs = QProgressBar()
        layout.addWidget(self.prog_3dgs)
        
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Start Gausfer Pipeline Training")
        self.btn_train.setMinimumHeight(50)
        self.btn_train.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_train.clicked.connect(self.start_training)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        layout.addStretch()

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov)")
        if file_name:
            self.video_path = file_name
            self.lbl_file.setText(os.path.basename(file_name))

    def select_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder_name:
            self.custom_image_dir = folder_name
            self.lbl_folder.setText(folder_name)

    def start_extraction(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video file first.")
            return
            
        self.btn_extract.setEnabled(False)
        self.prog_extraction.setValue(0)
        
        self.extractor_thread = VideoExtractorThread(self.video_path, self.spin_fps.value(), self.slider_blur.value())
        self.extractor_thread.progress.connect(self.prog_extraction.setValue)
        self.extractor_thread.log.connect(self.append_log)
        self.extractor_thread.error.connect(self.show_error)
        self.extractor_thread.finished_extraction.connect(self.on_extraction_finished)
        self.extractor_thread.start()

    def on_extraction_finished(self, saved_count):
        self.btn_extract.setEnabled(True)
        QMessageBox.information(self, "Success", f"Extracted {saved_count} frames successfully!")

    def stop_training(self):
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.stop()
            self.lbl_status.setText("Status: Stopping...")
            self.btn_stop.setEnabled(False)

    def start_training(self):
        # Reset bars
        self.prog_colmap.setValue(0)
        self.prog_nerf.setValue(0)
        self.prog_3dgs.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Status: Pipeline Running...")
        
        epochs = self.spin_epochs.value()
        steps = self.spin_steps.value()
        is_resume = self.chk_resume.isChecked()
        self.training_thread = PipelineTrainingThread(
            image_dir=self.custom_image_dir,
            num_epochs=epochs,
            num_steps=steps,
            resume_checkpoint=is_resume
        )
        self.training_thread.progress_colmap.connect(self.prog_colmap.setValue)
        self.training_thread.progress_nerf.connect(self.prog_nerf.setValue)
        self.training_thread.progress_3dgs.connect(self.prog_3dgs.setValue)
        self.training_thread.log.connect(self.append_log)
        self.training_thread.eta.connect(self.lbl_eta.setText)
        self.training_thread.error.connect(self.show_error)
        self.training_thread.finished_training.connect(self.on_training_finished)
        self.training_thread.start()

    def setup_visualizer_tab(self):
        layout = QVBoxLayout(self.tab_viewer)
        
        # Header / Controls
        ctrl_layout = QHBoxLayout()
        btn_load = QPushButton("Load Last Reconstruction")
        btn_load.clicked.connect(self.load_last_model)
        btn_load.setMinimumHeight(35)
        
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self.reset_viewer_camera)
        
        ctrl_layout.addWidget(btn_load)
        ctrl_layout.addWidget(btn_reset)
        
        # Scale Slider
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("Gaussian Scale:"))
        self.slider_gauss_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_gauss_scale.setRange(1, 100)
        self.slider_gauss_scale.setValue(100)
        self.slider_gauss_scale.setFixedWidth(150)
        self.slider_gauss_scale.valueChanged.connect(self.update_gauss_scale)
        ctrl_layout.addWidget(self.slider_gauss_scale)
        
        layout.addLayout(ctrl_layout)
        
        # Visualizer Widget
        self.visualizer = GaussianVisualizer()
        layout.addWidget(self.visualizer, stretch=1)
        
        # Help text
        help_label = QLabel("Controls: [Left Mouse] Orbit | [Right Mouse] Pan | [Scroll] Zoom")
        help_label.setStyleSheet("color: #888; font-size: 11px;")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(help_label)
        
        self.visualizer.start()

    def reset_viewer_camera(self):
        self.visualizer.camera.yaw = 0.0
        self.visualizer.camera.pitch = 0.0
        self.visualizer.camera.distance = 5.0
        self.visualizer.camera.target = torch.tensor([0.0, 0.0, 0.0], device='cuda')
        self.visualizer.needs_render = True
        self.visualizer.update()

    def load_last_model(self):
        path = "./output/gauss_checkpoint.pth"
        if os.path.exists(path):
            self.append_log(f"Loading local model for visualizer: {path}")
            if self.visualizer.load_model(path):
                self.append_log("Model loaded successfully!")
            else:
                self.append_log("Failed to parse model file.")
        else:
            QMessageBox.warning(self, "No Model", "No reconstruction checkpoint found in ./output/")

    def on_training_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Status: Reconstruction Finished!")
        # Auto-load into visualizer
        self.load_last_model()
        QMessageBox.information(self, "Training Complete", "Full Room Reconstruction completed. Interactive view is now ready.")

    def show_error(self, message):
        self.btn_extract.setEnabled(True)
        self.btn_train.setEnabled(True)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Pipeline Error", message)

    def append_log(self, text):
        self.console.appendPlainText(text)
        # Scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def redirect_stdout(self):
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.append_log)
        sys.stdout = self.stream
        # We can also redirect stderr to the console:
        # sys.stderr = self.stream

    def update_gauss_scale(self, value):
        self.visualizer.scale_modifier = value / 100.0
        self.visualizer.needs_render = True
        self.visualizer.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
