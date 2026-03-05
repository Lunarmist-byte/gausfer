import sys
import os
import cv2
import tempfile
import torch
import torch.optim as optim
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QProgressBar, QSlider, QSpinBox, QFileDialog, 
                             QPlainTextEdit, QMessageBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from pose_estimator import PoseEstimator
from dataset_loader import RoomDatasetLoader
from nerf_model import RoomNeRF
from nerf_trainer import NeRFTrainer
from bridge_converter import NeRFToGaussianBridge
from HybridCoOptimizer import HybridCoOptimizer
from rasterizer import RoomRasterizerCUDA
from gaussian_model import GaussianModel
from main import ViewCamera

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

class PipelineTrainingThread(QThread):
    progress_colmap = pyqtSignal(int)
    progress_nerf = pyqtSignal(int)
    progress_3dgs = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_training = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            image_dir = './images'
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

            # 2. Phase: NeRF Warmup
            self.progress_nerf.emit(5)
            self.log.emit("Loading Dataset...")
            dataset = RoomDatasetLoader(colmap_dir=colmap_model_path, images_dir=image_dir)
            
            self.log.emit("Initializing NeRF...")
            nerf = RoomNeRF().cuda()
            trainer = NeRFTrainer(nerf)
            
            self.log.emit("Starting NeRF Warmup (10 Epochs)...")
            num_epochs = 10
            for epoch in range(num_epochs):
                for i in range(len(dataset.images)):
                    img, poses = dataset.get_training_batch(i)
                    loss = trainer.train_step(img, poses)
                self.log.emit(f"NeRF Epoch {epoch} Loss: {loss:.4f}")
                self.progress_nerf.emit(int(((epoch + 1) / num_epochs) * 100))
            
            # 3. Phase: Bridging and 3DGS
            self.progress_3dgs.emit(5)
            self.log.emit("Seeding Gaussians via Neural Density...")
            bbox = [[-5, -5, -5], [5, 5, 5]]
            bridge = NeRFToGaussianBridge(nerf, bbox)
            init_xyz, init_rgb = bridge.generate_initial_gaussians()
            
            gaussians = GaussianModel(sh_degree=3)
            gaussians.initialize_from_nerf(init_xyz, init_rgb)
            
            gaussians_optim = optim.Adam(gaussians.parameters(), lr=0.001)
            self.log.emit("Starting Hybrid Co-Optimization...")
            co_optimizer = HybridCoOptimizer(nerf, gaussians)
            rasterizer = RoomRasterizerCUDA()

            num_steps = 5000
            for step in range(num_steps):
                gaussians_optim.zero_grad()
                idx = step % len(dataset.images)
                ground_truth_image, (H, W, K, c2w) = dataset.get_training_batch(idx)
                
                view_cam = ViewCamera(H, W, K, c2w)
                old_count = gaussians.xyz.shape[0]
                loss = co_optimizer.step(view_cam=view_cam, ground_truth_image=ground_truth_image, render_func=rasterizer.render_room_view)
                
                if gaussians.xyz.shape[0] != old_count:
                    gaussians_optim = optim.Adam(gaussians.parameters(), lr=0.001)

                gaussians_optim.step()
                
                if step % 100 == 0:
                    self.log.emit(f"Step {step}/{num_steps} | Loss: {loss:.4f} | Splat Count: {gaussians.xyz.shape[0]}")
                
                # Update 3DGS Progress
                if step % 50 == 0:
                    self.progress_3dgs.emit(int((step / num_steps) * 100))
            
            self.progress_3dgs.emit(100)
            self.log.emit("\n Full Room Reconstruction Done!")
            self.finished_training.emit()

        except Exception as e:
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
        self.tabs.addTab(self.tab_extraction, "1. Video Extraction")
        self.tabs.addTab(self.tab_training, "2. Execution Pipeline")
        main_layout.addWidget(self.tabs)
        
        self.setup_extraction_tab()
        self.setup_training_tab()
        
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
        
        lbl_desc = QLabel("Upload a room walkthrough video. The system will automatically extract sharp frames for the dataset.")
        layout.addWidget(lbl_desc)
        
        # Video Selection
        file_layout = QHBoxLayout()
        self.lbl_file = QLabel("No Video Selected")
        btn_select = QPushButton("Select Video")
        btn_select.clicked.connect(self.select_video)
        file_layout.addWidget(self.lbl_file)
        file_layout.addWidget(btn_select)
        layout.addLayout(file_layout)
        
        # Settings
        settings_layout = QHBoxLayout()
        # FPS
        fps_layout = QVBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 10)
        self.spin_fps.setValue(2)
        fps_layout.addWidget(self.spin_fps)
        settings_layout.addLayout(fps_layout)
        
        # Blur Strictness
        blur_layout = QVBoxLayout()
        blur_layout.addWidget(QLabel("Blur Strictness Threshold:"))
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(50, 300)
        self.slider_blur.setValue(100)
        blur_layout.addWidget(self.slider_blur)
        settings_layout.addLayout(blur_layout)
        
        layout.addLayout(settings_layout)
        
        # Extraction control
        self.btn_extract = QPushButton("Start Extraction")
        self.btn_extract.setMinimumHeight(40)
        self.btn_extract.clicked.connect(self.start_extraction)
        layout.addWidget(self.btn_extract)
        
        self.prog_extraction = QProgressBar()
        layout.addWidget(self.prog_extraction)
        layout.addStretch()

        self.video_path = None

    def setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)
        
        lbl_desc = QLabel("Execute the Hybrid Gausfer NeRF+3DGS Pipeline.")
        layout.addWidget(lbl_desc)
        
        # Status Label
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_status)
        
        # COLMAP Progress
        layout.addWidget(QLabel("Phase 1: COLMAP Pose Estimation"))
        self.prog_colmap = QProgressBar()
        layout.addWidget(self.prog_colmap)
        
        # NeRF Progress
        layout.addWidget(QLabel("Phase 2: NeRF Warmup (10 Epochs)"))
        self.prog_nerf = QProgressBar()
        layout.addWidget(self.prog_nerf)
        
        # 3DGS Progress
        layout.addWidget(QLabel("Phase 3: Hybrid Co-Optimization (5000 Steps)"))
        self.prog_3dgs = QProgressBar()
        layout.addWidget(self.prog_3dgs)
        
        self.btn_train = QPushButton("Start Gausfer Pipeline Training")
        self.btn_train.setMinimumHeight(50)
        self.btn_train.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)
        
        layout.addStretch()

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov)")
        if file_name:
            self.video_path = file_name
            self.lbl_file.setText(os.path.basename(file_name))

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

    def start_training(self):
        # Reset bars
        self.prog_colmap.setValue(0)
        self.prog_nerf.setValue(0)
        self.prog_3dgs.setValue(0)
        self.btn_train.setEnabled(False)
        self.lbl_status.setText("Status: Pipeline Running...")
        
        self.training_thread = PipelineTrainingThread()
        self.training_thread.progress_colmap.connect(self.prog_colmap.setValue)
        self.training_thread.progress_nerf.connect(self.prog_nerf.setValue)
        self.training_thread.progress_3dgs.connect(self.prog_3dgs.setValue)
        self.training_thread.log.connect(self.append_log)
        self.training_thread.error.connect(self.show_error)
        self.training_thread.finished_training.connect(self.on_training_finished)
        self.training_thread.start()

    def on_training_finished(self):
        self.btn_train.setEnabled(True)
        self.lbl_status.setText("Status: Reconstruction Finished! 🎯")
        QMessageBox.information(self, "Training Complete", "Full Room Reconstruction completed.")

    def show_error(self, message):
        self.btn_extract.setEnabled(True)
        self.btn_train.setEnabled(True)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
