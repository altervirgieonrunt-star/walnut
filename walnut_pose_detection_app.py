import sys
import os

# Linux specific configuration for Qt
if sys.platform.startswith('linux'):
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QSpinBox, QGroupBox, QGridLayout, QTextEdit,
                             QComboBox, QCheckBox, QTabWidget, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont


import cv2

# Fix for opencv-python-headless interacting with Qt on Linux
if sys.platform.startswith('linux'):
    cv2_plugin_path = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins')
    if 'QT_PLUGIN_PATH' in os.environ:
        paths = os.environ['QT_PLUGIN_PATH'].split(':')
        paths = [p for p in paths if cv2_plugin_path not in p]
        os.environ['QT_PLUGIN_PATH'] = ':'.join(paths)

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json


class PoseDetectionThread(QThread):

    result_ready = pyqtSignal(object, object, object)  # image, results, pose_info
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.image = None
        self.track_line = None
        self.angle_threshold = 15
        self.collinearity_threshold = 10
        self.running = False
        
    def set_model(self, model):
        self.model = model
        
    def set_image(self, image):
        self.image = image.copy()
        
    def set_track_line(self, track_line):
        self.track_line = track_line
        
    def set_thresholds(self, angle_threshold, collinearity_threshold):
        self.angle_threshold = angle_threshold
        self.collinearity_threshold = collinearity_threshold
        
    def run(self):
        if self.model is None or self.image is None:
            return
            

        results = self.model(self.image, verbose=False, conf=0.25)
        

        pose_info = self.calculate_pose(results[0], self.track_line)
        
        self.result_ready.emit(self.image, results[0], pose_info)
        
    def calculate_pose(self, result, track_line):
        pose_info = []
        
        if result.keypoints is None or len(result.keypoints) == 0:
            return pose_info
            
        keypoints = result.keypoints.data.cpu().numpy()
        
        for kpt in keypoints:
            if len(kpt) < 3:
                continue
            
            if kpt[0][2] < 0.5 or kpt[1][2] < 0.5 or kpt[2][2] < 0.5:
                continue
                
            up = kpt[0][:2]  # up point (x, y)
            side = kpt[1][:2]  # side point (x, y)
            bottom = kpt[2][:2]  # bottom point (x, y)
            
            collinearity = self.calculate_collinearity(up, side, bottom)
            
            walnut_angle = self.calculate_angle(up, bottom)
            
            if track_line is not None:
                track_angle = self.calculate_angle(track_line[0], track_line[1])
                angle_diff = abs(walnut_angle - track_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
            else:
                track_angle = None
                angle_diff = None
            
            is_aligned = True
            reasons = []
            
            if collinearity > self.collinearity_threshold:
                is_aligned = False
                reasons.append(f"三点不共线(偏差:{collinearity:.1f}px)")
                
            if angle_diff is not None and angle_diff > self.angle_threshold:
                is_aligned = False
                reasons.append(f"角度偏差过大({angle_diff:.1f}°)")
            
            pose_info.append({
                'keypoints': kpt,
                'up': up,
                'side': side,
                'bottom': bottom,
                'collinearity': collinearity,
                'walnut_angle': walnut_angle,
                'track_angle': track_angle,
                'angle_diff': angle_diff,
                'is_aligned': is_aligned,
                'reasons': reasons
            })
            
        return pose_info
    
    @staticmethod
    def calculate_collinearity(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        numerator = abs((y3 - y1) * x2 - (x3 - x1) * y2 + x3 * y1 - y3 * x1)
        denominator = np.sqrt((y3 - y1)**2 + (x3 - x1)**2)
        
        if denominator == 0:
            return 0
            
        distance = numerator / denominator
        return distance
    
    @staticmethod
    def calculate_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle


class WalnutPoseDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("核桃姿态识别检测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.model = None
        self.current_image = None
        self.track_line = None
        self.calibration_points = []
        self.is_calibrating = False
        self.camera_active = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.image_list = []
        self.current_image_index = 0
        self.folder_path = None
        self.batch_results = {}
        
        self.angle_threshold = 15
        self.collinearity_threshold = 10
        
        self.init_ui()
        
        self.detection_thread = PoseDetectionThread()
        self.detection_thread.result_ready.connect(self.on_detection_complete)
        
        
        self.load_config()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        display_area = self.create_display_area()
        display_area = self.create_display_area()
        splitter.addWidget(display_area)
        

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        model_group = QGroupBox("模型加载")
        model_layout = QVBoxLayout()
        
        self.model_path_label = QLabel("未加载模型")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label)
        
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        calibration_group = QGroupBox("轨道标定")
        calibration_layout = QVBoxLayout()
        
        self.calibration_status_label = QLabel("未标定")
        calibration_layout.addWidget(self.calibration_status_label)
        
        self.calibrate_btn = QPushButton("开始标定（点击图像选择2点）")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        calibration_layout.addWidget(self.calibrate_btn)
        
        clear_calibration_btn = QPushButton("清除标定")
        clear_calibration_btn.clicked.connect(self.clear_calibration)
        calibration_layout.addWidget(clear_calibration_btn)
        
        save_calibration_btn = QPushButton("保存标定")
        save_calibration_btn.clicked.connect(self.save_calibration)
        calibration_layout.addWidget(save_calibration_btn)
        
        load_calibration_btn = QPushButton("加载标定")
        load_calibration_btn.clicked.connect(self.load_calibration)
        calibration_layout.addWidget(load_calibration_btn)
        
        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)
        
        params_group = QGroupBox("检测参数")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("角度阈值(°):"), 0, 0)
        self.angle_threshold_spin = QSpinBox()
        self.angle_threshold_spin.setRange(1, 90)
        self.angle_threshold_spin.setValue(self.angle_threshold)
        self.angle_threshold_spin.valueChanged.connect(self.update_thresholds)
        params_layout.addWidget(self.angle_threshold_spin, 0, 1)
        
        params_layout.addWidget(QLabel("共线阈值(px):"), 1, 0)
        self.collinearity_threshold_spin = QSpinBox()
        self.collinearity_threshold_spin.setRange(1, 50)
        self.collinearity_threshold_spin.setValue(self.collinearity_threshold)
        self.collinearity_threshold_spin.valueChanged.connect(self.update_thresholds)
        params_layout.addWidget(self.collinearity_threshold_spin, 1, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        input_group = QGroupBox("输入源")
        input_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("加载图片")
        self.load_image_btn.clicked.connect(self.load_single_image)
        input_layout.addWidget(self.load_image_btn)
        
        self.load_folder_btn = QPushButton("加载文件夹")
        self.load_folder_btn.clicked.connect(self.load_image_folder)
        input_layout.addWidget(self.load_folder_btn)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        input_layout.addLayout(nav_layout)
        
        self.image_index_label = QLabel("0 / 0")
        self.image_index_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(self.image_index_label)
        
        self.camera_btn = QPushButton("启动摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        input_layout.addWidget(self.camera_btn)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2"])
        input_layout.addWidget(self.camera_combo)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        detect_group = QGroupBox("检测控制")
        detect_layout = QVBoxLayout()
        
        self.detect_btn = QPushButton("执行检测")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        detect_layout.addWidget(self.detect_btn)
        
        self.batch_detect_btn = QPushButton("批量检测文件夹")
        self.batch_detect_btn.clicked.connect(self.batch_detect_folder)
        self.batch_detect_btn.setEnabled(False)
        detect_layout.addWidget(self.batch_detect_btn)
        
        self.save_results_btn = QPushButton("保存检测结果")
        self.save_results_btn.clicked.connect(self.save_detection_results)
        self.save_results_btn.setEnabled(False)
        detect_layout.addWidget(self.save_results_btn)
        
        self.auto_detect_checkbox = QCheckBox("切换图片时自动检测")
        self.auto_detect_checkbox.setChecked(True)
        detect_layout.addWidget(self.auto_detect_checkbox)
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        layout.addStretch()
        
        return panel
    
    def create_display_area(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #000;")
        self.image_label.clicked.connect(self.on_image_clicked)
        layout.addWidget(self.image_label)
        
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        result_layout.addWidget(self.result_text)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        return widget
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "Model Files (*.pt *.yaml);;All Files (*)"
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.detection_thread.set_model(self.model)
                self.model_path_label.setText(f"已加载: {os.path.basename(file_path)}")
                self.detect_btn.setEnabled(True)
                
                # 如果已经加载了文件夹，启用批量检测按钮
                if self.image_list:
                    self.batch_detect_btn.setEnabled(True)
                
                self.log_message(f"模型加载成功: {file_path}")
            except Exception as e:
                self.log_message(f"模型加载失败: {e}")
    
    def start_calibration(self):
        if self.current_image is None:
            self.log_message("请先加载图像")
            return
            
        self.is_calibrating = True
        self.calibration_points = []
        self.calibrate_btn.setText("标定中... (已选择0/2点)")
        self.log_message("请在图像上点击两个点定义轨道方向")
    
    def clear_calibration(self):
        self.track_line = None
        self.calibration_points = []
        self.is_calibrating = False
        self.calibrate_btn.setText("开始标定（点击图像选择2点）")
        self.calibration_status_label.setText("未标定")
        self.detection_thread.set_track_line(None)
        self.log_message("标定已清除")
        self.update_display()
    
    def save_calibration(self):
        if self.track_line is None:
            self.log_message("没有可保存的标定数据")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标定文件", "calibration.json", "JSON Files (*.json)"
        )
        
        if file_path:
            data = {
                'track_line': [list(p) for p in self.track_line],
                'angle_threshold': self.angle_threshold,
                'collinearity_threshold': self.collinearity_threshold
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.log_message(f"标定已保存: {file_path}")
    
    def load_calibration(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载标定文件", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.track_line = [tuple(p) for p in data['track_line']]
                self.angle_threshold = data.get('angle_threshold', 15)
                self.collinearity_threshold = data.get('collinearity_threshold', 10)
                
                self.angle_threshold_spin.setValue(self.angle_threshold)
                self.collinearity_threshold_spin.setValue(self.collinearity_threshold)
                
                self.detection_thread.set_track_line(self.track_line)
                
                angle = PoseDetectionThread.calculate_angle(self.track_line[0], self.track_line[1])
                self.calibration_status_label.setText(f"已标定 (角度: {angle:.1f}°)")
                self.log_message(f"标定已加载: {file_path}")
                self.update_display()
            except Exception as e:
                self.log_message(f"标定加载失败: {e}")
    
    def on_image_clicked(self, x, y):
        if not self.is_calibrating or self.current_image is None:
            return
            
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
            
        img_h, img_w = self.current_image.shape[:2]
        scale_x = img_w / pixmap.width()
        scale_y = img_h / pixmap.height()
        
        offset_x = (label_size.width() - pixmap.width()) / 2
        offset_y = (label_size.height() - pixmap.height()) / 2
        
        img_x = int((x - offset_x) * scale_x)
        img_y = int((y - offset_y) * scale_y)
        
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        
        self.calibration_points.append((img_x, img_y))
        self.calibrate_btn.setText(f"标定中... (已选择{len(self.calibration_points)}/2点)")
        
        if len(self.calibration_points) == 2:
            self.track_line = self.calibration_points.copy()
            self.is_calibrating = False
            self.calibrate_btn.setText("开始标定（点击图像选择2点）")
            
            angle = PoseDetectionThread.calculate_angle(self.track_line[0], self.track_line[1])
            self.calibration_status_label.setText(f"已标定 (角度: {angle:.1f}°)")
            self.detection_thread.set_track_line(self.track_line)
            self.log_message(f"轨道标定完成，角度: {angle:.1f}°")
        
        self.update_display()
    
    def update_thresholds(self):
        self.angle_threshold = self.angle_threshold_spin.value()
        self.collinearity_threshold = self.collinearity_threshold_spin.value()
        self.detection_thread.set_thresholds(self.angle_threshold, self.collinearity_threshold)
        
        if self.auto_detect_checkbox.isChecked() and self.current_image is not None:
            self.run_detection()
    
    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.image_list = [file_path]
            self.current_image_index = 0
            self.load_image_at_index(0)
            self.update_navigation_buttons()
    
    def load_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        
        if folder_path:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
            self.image_list = []
            
            for ext in extensions:
                self.image_list.extend(Path(folder_path).glob(f'*{ext}'))
            
            self.image_list = sorted([str(p) for p in self.image_list])
            self.folder_path = folder_path  # 保存文件夹路径
            
            if self.image_list:
                self.current_image_index = 0
                self.load_image_at_index(0)
                self.update_navigation_buttons()
                
                if self.model is not None:
                    self.batch_detect_btn.setEnabled(True)
                
                self.log_message(f"加载了 {len(self.image_list)} 张图片")
                self.log_message(f"文件夹: {folder_path}")
                
                # 提示批量处理
                if len(self.image_list) > 1 and self.model is not None:
                    self.log_message(f"提示: 可以使用'批量检测文件夹'按钮处理所有图片")
            else:
                self.log_message("文件夹中没有找到图片")
    
    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_list):
            image_path = self.image_list[index]
            self.current_image = cv2.imread(image_path)  # Keep as BGR
            if self.current_image is not None:
                self.update_display()
                
                if self.auto_detect_checkbox.isChecked() and self.model is not None:
                    self.run_detection()
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image_at_index(self.current_image_index)
            self.update_navigation_buttons()
    
    def next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image_at_index(self.current_image_index)
            self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        has_images = len(self.image_list) > 0
        self.prev_btn.setEnabled(has_images and self.current_image_index > 0)
        self.next_btn.setEnabled(has_images and self.current_image_index < len(self.image_list) - 1)
        self.image_index_label.setText(f"{self.current_image_index + 1} / {len(self.image_list)}")
    
    def toggle_camera(self):
        if not self.camera_active:
            camera_index = self.camera_combo.currentIndex()
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                self.camera_active = True
                self.camera_btn.setText("停止摄像头")
                self.timer.start(30)  # 30ms更新一次
                self.log_message(f"摄像头 {camera_index} 已启动")
            else:
                self.log_message(f"无法打开摄像头 {camera_index}")
        else:
            self.stop_camera()
    
    def stop_camera(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.camera_active = False
            self.camera_btn.setText("启动摄像头")
            self.log_message("摄像头已停止")
    
    def update_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame  # Keep as BGR
                
                if self.auto_detect_checkbox.isChecked() and self.model is not None:
                    self.run_detection()
                else:
                    self.update_display()
    
    def run_detection(self):
        if self.model is None:
            self.log_message("请先加载模型")
            return
            
        if self.current_image is None:
            self.log_message("请先加载图像或启动摄像头")
            return
        
        # 使用线程进行检测
        if not self.detection_thread.isRunning():
            self.detection_thread.set_image(self.current_image)
            self.detection_thread.set_track_line(self.track_line)
            self.detection_thread.set_thresholds(self.angle_threshold, self.collinearity_threshold)
            self.detection_thread.start()
    
    def on_detection_complete(self, image, results, pose_info):
        annotated_image = self.draw_results(image.copy(), results, pose_info)
        self.current_image = annotated_image
        self.update_display()
        
        self.display_results(pose_info)
    
    def draw_results(self, image, results, pose_info):
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                conf_text = f"Walnut {conf:.2f}"
                cv2.putText(image, conf_text, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for info in pose_info:
            kpt = info['keypoints']
            up, side, bottom = info['up'], info['side'], info['bottom']
            is_aligned = info['is_aligned']
            
            color_up = (255, 0, 0)  # 红色 - up
            color_side = (0, 255, 0)  # 绿色 - side
            color_bottom = (0, 0, 255)  # 蓝色 - bottom
            
            cv2.circle(image, (int(up[0]), int(up[1])), 5, color_up, -1)
            cv2.circle(image, (int(side[0]), int(side[1])), 5, color_side, -1)
            cv2.circle(image, (int(bottom[0]), int(bottom[1])), 5, color_bottom, -1)
            
            line_color = (0, 255, 0) if is_aligned else (255, 0, 0)
            cv2.line(image, (int(up[0]), int(up[1])), (int(bottom[0]), int(bottom[1])), line_color, 2)
            
            status_text = "OK" if is_aligned else "NG"
            status_color = (0, 255, 0) if is_aligned else (255, 0, 0)
            cv2.putText(image, status_text, (int(up[0]) + 10, int(up[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if info['angle_diff'] is not None:
                angle_text = f"Angle: {info['angle_diff']:.1f}deg"
                cv2.putText(image, angle_text, (int(up[0]) + 10, int(up[1]) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            collinearity_text = f"Dev: {info['collinearity']:.1f}px"
            cv2.putText(image, collinearity_text, (int(up[0]) + 10, int(up[1]) + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.track_line is not None:
            cv2.line(image, self.track_line[0], self.track_line[1], (255, 255, 0), 3)
            cv2.circle(image, self.track_line[0], 8, (255, 255, 0), -1)
            cv2.circle(image, self.track_line[1], 8, (255, 255, 0), -1)
        
        for i, pt in enumerate(self.calibration_points):
            cv2.circle(image, pt, 8, (0, 255, 255), -1)
            cv2.putText(image, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return image
    
    def display_results(self, pose_info):
        if not pose_info:
            self.result_text.setText("未检测到核桃")
            return
        
        text = f"检测到 {len(pose_info)} 个核桃:\n\n"
        
        for i, info in enumerate(pose_info):
            text += f"核桃 #{i+1}:\n"
            text += f"  - 共线偏差: {info['collinearity']:.2f} px\n"
            text += f"  - 核桃角度: {info['walnut_angle']:.2f}°\n"
            
            if info['track_angle'] is not None:
                text += f"  - 轨道角度: {info['track_angle']:.2f}°\n"
                text += f"  - 角度差: {info['angle_diff']:.2f}°\n"
            
            text += f"  - 姿态: {'正确' if info['is_aligned'] else '错误'}\n"
            
            if not info['is_aligned']:
                text += f"  - 原因: {', '.join(info['reasons'])}\n"
            
            text += "\n"
        
        self.result_text.setText(text)
    
    def update_display(self):
        if self.current_image is not None:
            # Convert BGR→RGB only for Qt display
            display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            h, w, ch = display_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
    
    def log_message(self, message):
        current_text = self.result_text.toPlainText()
        self.result_text.setText(f"{message}\n{'-'*50}\n{current_text}")
    
    def save_config(self):
        config = {
            'angle_threshold': self.angle_threshold,
            'collinearity_threshold': self.collinearity_threshold,
        }
        with open('walnut_pose_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def batch_detect_folder(self):
        if self.model is None:
            self.log_message("请先加载模型")
            return
        
        if not self.image_list:
            self.log_message("请先加载图片文件夹")
            return
        
        self.log_message(f"开始批量检测 {len(self.image_list)} 张图片...")
        self.batch_results = {}
        
        ok_count = 0
        ng_count = 0
        no_detection_count = 0
        
        for i, image_path in enumerate(self.image_list):    
            image = cv2.imread(image_path)
            if image is None:
                self.log_message(f"无法读取: {os.path.basename(image_path)}")
                continue
            
            # Keep as BGR — YOLO handles BGR→RGB internally
            results = self.model(image, verbose=False, conf=0.25)
            
            detection_thread = PoseDetectionThread()
            detection_thread.set_thresholds(self.angle_threshold, self.collinearity_threshold)
            pose_info = detection_thread.calculate_pose(results[0], self.track_line)
            
            self.batch_results[image_path] = {
                'pose_info': pose_info,
                'results': results[0]
            }
            
            if len(pose_info) == 0:
                no_detection_count += 1
            else:
                for info in pose_info:
                    if info['is_aligned']:
                        ok_count += 1
                    else:
                        ng_count += 1
            
            if (i + 1) % 10 == 0 or i == len(self.image_list) - 1:
                self.log_message(f"进度: {i + 1}/{len(self.image_list)}")
        
        self.log_message("=" * 50)
        self.log_message(f"批量检测完成!")
        self.log_message(f"总图片数: {len(self.image_list)}")
        self.log_message(f"检测到核桃: {ok_count + ng_count}")
        self.log_message(f"姿态OK: {ok_count}")
        self.log_message(f"姿态NG: {ng_count}")
        self.log_message(f"未检测到: {no_detection_count}")
        self.log_message(f"OK率: {ok_count/(ok_count + ng_count)*100:.1f}%" if (ok_count + ng_count) > 0 else "N/A")
        self.log_message("=" * 50)
        
        self.save_results_btn.setEnabled(True)
    
    def save_detection_results(self):
        if not self.batch_results:
            self.log_message("没有检测结果可保存")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "detection_results.txt", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not save_path:
            return
        
        try:
            if save_path.endswith('.csv'):
                self.save_results_csv(save_path)
            else:
                self.save_results_txt(save_path)
            
            self.log_message(f"结果已保存: {save_path}")
        except Exception as e:
            self.log_message(f"保存失败: {e}")
    
    def save_results_txt(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("核桃姿态检测结果报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"检测参数:\n")
            f.write(f"  角度阈值: {self.angle_threshold}°\n")
            f.write(f"  共线阈值: {self.collinearity_threshold}px\n")
            if self.track_line:
                track_angle = PoseDetectionThread.calculate_angle(self.track_line[0], self.track_line[1])
                f.write(f"  轨道角度: {track_angle:.1f}°\n")
            f.write("\n")
            
            ok_count = 0
            ng_count = 0
            
            for image_path, data in self.batch_results.items():
                pose_info = data['pose_info']
                f.write("-" * 80 + "\n")
                f.write(f"图片: {os.path.basename(image_path)}\n")
                
                if len(pose_info) == 0:
                    f.write("  结果: 未检测到核桃\n")
                else:
                    for i, info in enumerate(pose_info):
                        f.write(f"\n  核桃 #{i+1}:\n")
                        f.write(f"    姿态: {'OK' if info['is_aligned'] else 'NG'}\n")
                        f.write(f"    共线偏差: {info['collinearity']:.2f}px\n")
                        f.write(f"    核桃角度: {info['walnut_angle']:.2f}°\n")
                        if info['angle_diff'] is not None:
                            f.write(f"    角度差: {info['angle_diff']:.2f}°\n")
                        
                        if not info['is_aligned']:
                            f.write(f"    错误原因: {', '.join(info['reasons'])}\n")
                        
                        if info['is_aligned']:
                            ok_count += 1
                        else:
                            ng_count += 1
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("统计汇总:\n")
            f.write(f"  总图片数: {len(self.batch_results)}\n")
            f.write(f"  姿态OK: {ok_count}\n")
            f.write(f"  姿态NG: {ng_count}\n")
            if (ok_count + ng_count) > 0:
                f.write(f"  OK率: {ok_count/(ok_count + ng_count)*100:.1f}%\n")
            f.write("=" * 80 + "\n")
    
    def save_results_csv(self, save_path):
        import csv
        
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['图片名称', '核桃编号', '姿态', '共线偏差(px)', '核桃角度(°)', '角度差(°)', '错误原因'])
            
            for image_path, data in self.batch_results.items():
                pose_info = data['pose_info']
                image_name = os.path.basename(image_path)
                
                if len(pose_info) == 0:
                    writer.writerow([image_name, '-', '未检测到', '-', '-', '-', '-'])
                else:
                    for i, info in enumerate(pose_info):
                        writer.writerow([
                            image_name,
                            i + 1,
                            'OK' if info['is_aligned'] else 'NG',
                            f"{info['collinearity']:.2f}",
                            f"{info['walnut_angle']:.2f}",
                            f"{info['angle_diff']:.2f}" if info['angle_diff'] is not None else '-',
                            ', '.join(info['reasons']) if not info['is_aligned'] else '-'
                        ])
    
    def load_config(self):
        try:
            if os.path.exists('walnut_pose_config.json'):
                with open('walnut_pose_config.json', 'r') as f:
                    config = json.load(f)
                self.angle_threshold = config.get('angle_threshold', 15)
                self.collinearity_threshold = config.get('collinearity_threshold', 10)
        except:
            pass
    
    def closeEvent(self, event):
        self.stop_camera()
        self.save_config()
        event.accept()


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())


def main():
    app = QApplication(sys.argv)
    window = WalnutPoseDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

