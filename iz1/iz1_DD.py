import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import time
import json
from collections import defaultdict
import math

class CustomCSRTTracker:
    def __init__(self):
        self.initialized = False
        self.bbox = None
        self.window_size = 30
        self.learning_rate = 0.025
        self.hist_bins = 16
        self.color_spaces = ['hsv', 'lab']
        self.object_roi = None
        self.object_hists = {}
        self.object_template = None
        self.obj_width = 0
        self.obj_height = 0

    def init(self, frame, bbox):
        self.bbox = bbox
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            return False
        self.object_roi = frame[y:y+h, x:x+w].copy()
        if self.object_roi.size == 0:
            return False
        self.object_hists = self._compute_histograms(self.object_roi)
        self.object_template = cv2.resize(self.object_roi, (32, 32)).astype(np.uint8)
        self.obj_width = w
        self.obj_height = h
        self.initialized = True
        return True

    def _compute_histograms(self, roi):
        hists = {}
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [self.hist_bins, self.hist_bins], [0, 180, 0, 256])
            cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
            hists['hsv'] = h
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l = cv2.calcHist([lab], [1, 2], None, [self.hist_bins, self.hist_bins], [0, 256, 0, 256])
            cv2.normalize(l, l, 0, 1, cv2.NORM_MINMAX)
            hists['lab'] = l
        except Exception:
            pass
        return hists

    def _compute_correlation(self, search_roi):
        try:
            if search_roi is None or search_roi.size == 0:
                return np.array([[-1]])
            template_resized = cv2.resize(self.object_template, (32, 32)).astype(np.uint8)
            search_resized = cv2.resize(search_roi, (64, 64)).astype(np.uint8)
            if search_resized.shape[0] < template_resized.shape[0] or search_resized.shape[1] < template_resized.shape[1]:
                return np.array([[-1]])
            correlation = cv2.matchTemplate(search_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            return correlation
        except Exception:
            return np.array([[-1]])

    def _compute_histogram_similarity(self, roi):
        similarity = 0.0
        weight = 1.0 / max(1, len(self.color_spaces))
        try:
            current_hists = self._compute_histograms(roi)
            for color_space in self.color_spaces:
                if color_space in self.object_hists and color_space in current_hists:
                    sim = cv2.compareHist(self.object_hists[color_space], current_hists[color_space], cv2.HISTCMP_CORREL)
                    similarity += sim * weight
        except Exception:
            pass
        return max(0.0, similarity)

    def update(self, frame):
        if not self.initialized:
            return False, (0, 0, 0, 0)
        x, y, w, h = [int(v) for v in self.bbox]
        search_x1 = max(0, x - self.window_size)
        search_y1 = max(0, y - self.window_size)
        search_x2 = min(frame.shape[1], x + w + self.window_size)
        search_y2 = min(frame.shape[0], y + h + self.window_size)
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            return False, self.bbox
        search_roi_full = frame[search_y1:search_y2, search_x1:search_x2].copy()
        if search_roi_full.size == 0:
            return False, self.bbox

        best_score = -1.0
        best_bbox = self.bbox
        step = max(2, self.window_size // 8)
        max_iterations = 400
        iterations = 0
        prev_cx = x + w / 2.0
        prev_cy = y + h / 2.0
        center_search_x = (search_roi_full.shape[1] - w) // 2
        center_search_y = (search_roi_full.shape[0] - h) // 2

        candidates = []
        for dy in range(0, max(1, search_roi_full.shape[0] - h + 1), step):
            for dx in range(0, max(1, search_roi_full.shape[1] - w + 1), step):
                cx = search_x1 + dx + w / 2.0
                cy = search_y1 + dy + h / 2.0
                dist_sq = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
                candidates.append((dist_sq, dx, dy))
        candidates.sort(key=lambda t: t[0])

        max_shift = max(w, h) * 3.0
        sigma = max(w, h) * 1.5
        spatial_weight = 0.6

        for dist_sq, dx, dy in candidates:
            iterations += 1
            if iterations > max_iterations:
                break
            candidate_roi = search_roi_full[dy:dy+h, dx:dx+w]
            if candidate_roi.size == 0 or candidate_roi.shape[0] != h or candidate_roi.shape[1] != w:
                continue
            try:
                correlation_map = self._compute_correlation(candidate_roi)
                correlation_score = float(np.max(correlation_map)) if correlation_map.size > 0 else -1.0
                hist_score = self._compute_histogram_similarity(candidate_roi)
                cand_cx = search_x1 + dx + w / 2.0
                cand_cy = search_y1 + dy + h / 2.0
                center_dist = math.hypot(cand_cx - prev_cx, cand_cy - prev_cy)
                if center_dist > max_shift:
                    continue
                spatial_prior = math.exp(- (center_dist ** 2) / (2.0 * (sigma ** 2) + 1e-6))
                total_score = 0.6 * correlation_score + 0.4 * hist_score
                total_score = total_score * ( (1.0 - spatial_weight) + spatial_weight * spatial_prior )
                if total_score > best_score:
                    best_score = total_score
                    best_bbox = (search_x1 + dx, search_y1 + dy, w, h)
            except Exception:
                continue

        if best_score > 0.35:
            self.bbox = best_bbox
            x_new, y_new, w_new, h_new = [int(v) for v in best_bbox]
            new_roi = frame[y_new:y_new+h_new, x_new:x_new+w_new].copy()
            if new_roi.size > 0 and new_roi.shape[0] == h_new and new_roi.shape[1] == w_new:
                try:
                    new_hists = self._compute_histograms(new_roi)
                    for color_space in self.color_spaces:
                        if color_space in self.object_hists and color_space in new_hists:
                            self.object_hists[color_space] = (1 - self.learning_rate) * self.object_hists[color_space] + self.learning_rate * new_hists[color_space]
                    new_template = cv2.resize(new_roi, (32, 32)).astype(np.uint8)
                    self.object_template = ((1 - self.learning_rate) * self.object_template.astype(np.float32) + self.learning_rate * new_template.astype(np.float32)).astype(np.uint8)
                except Exception:
                    pass
            return True, best_bbox
        else:
            return False, self.bbox

class ObjectTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сравнительный анализ методов трекинга - OpenCV 4.12.0")
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.window_width = int(self.screen_width * 0.9)
        self.window_height = int(self.screen_height * 0.9)
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(1200, 800)
        self.cap = None
        self.tracking = False
        self.selected_bbox = None
        self.current_frame = None
        self.trackers = {}
        self.tracker_types = {}
        self.performance_data = defaultdict(list)
        self.camera_mode = False
        self.camera_index = 0
        self.display_width = int(self.window_width * 0.18)
        self.display_height = int(self.display_width * 0.75)
        self.create_interface()
        self.init_trackers()

    def create_interface(self):
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        title_frame = tk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(title_frame, text="Сравнительный анализ методов трекинга объектов", font=("Arial", 14, "bold")).pack()
        #tk.Label(title_frame, text="Включая собственную реализацию CSRT", font=("Arial", 11)).pack()
        video_scroll_frame = tk.Frame(main_container)
        video_scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(video_scroll_frame, bg="white")
        scrollbar_x = ttk.Scrollbar(video_scroll_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scrollbar_y = ttk.Scrollbar(video_scroll_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_container = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.video_container, anchor="nw")
        self.video_container.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.video_frames = {}
        video_grid_frame = tk.Frame(self.video_container)
        video_grid_frame.pack(pady=10)
        row1_frame = tk.Frame(video_grid_frame)
        row1_frame.pack(pady=5)
        original_frame = tk.LabelFrame(row1_frame, text="Исходное видео", font=("Arial", 10, "bold"), width=self.display_width, height=self.display_height)
        original_frame.pack(side=tk.LEFT, padx=5, pady=5)
        original_frame.pack_propagate(False)
        self.video_frames['original'] = tk.Label(original_frame)
        self.video_frames['original'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        csrt_frame = tk.LabelFrame(row1_frame, text="CSRT Tracker (OpenCV)", font=("Arial", 10, "bold"), width=self.display_width, height=self.display_height)
        csrt_frame.pack(side=tk.LEFT, padx=5, pady=5)
        csrt_frame.pack_propagate(False)
        self.video_frames['csrt'] = tk.Label(csrt_frame)
        self.video_frames['csrt'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        kcf_frame = tk.LabelFrame(row1_frame, text="KCF Tracker (OpenCV)", font=("Arial", 10, "bold"), width=self.display_width, height=self.display_height)
        kcf_frame.pack(side=tk.LEFT, padx=5, pady=5)
        kcf_frame.pack_propagate(False)
        self.video_frames['kcf'] = tk.Label(kcf_frame)
        self.video_frames['kcf'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        row2_frame = tk.Frame(video_grid_frame)
        row2_frame.pack(pady=5)
        mosse_frame = tk.LabelFrame(row2_frame, text="MOSSE Tracker (OpenCV)", font=("Arial", 10, "bold"), width=self.display_width, height=self.display_height)
        mosse_frame.pack(side=tk.LEFT, padx=5, pady=5)
        mosse_frame.pack_propagate(False)
        self.video_frames['mosse'] = tk.Label(mosse_frame)
        self.video_frames['mosse'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        custom_frame = tk.LabelFrame(row2_frame, text="Custom CSRT (Наша реализация)", font=("Arial", 10, "bold"), width=self.display_width, height=self.display_height)
        custom_frame.pack(side=tk.LEFT, padx=5, pady=5)
        custom_frame.pack_propagate(False)
        self.video_frames['custom_csrt'] = tk.Label(custom_frame)
        self.video_frames['custom_csrt'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        source_frame = tk.LabelFrame(main_container, text="Источник видео", font=("Arial", 11, "bold"))
        source_frame.pack(fill=tk.X, pady=10)
        self.create_source_panel(source_frame)
        self.settings_frame = tk.LabelFrame(main_container, text="Настройки трекинга", font=("Arial", 11, "bold"))
        self.settings_frame.pack(fill=tk.X, pady=10)
        self.create_settings_panel()
        self.control_frame = tk.Frame(main_container)
        self.control_frame.pack(fill=tk.X, pady=10)
        self.create_control_panel()
        self.info_frame = tk.LabelFrame(main_container, text="Информация", font=("Arial", 10))
        self.info_frame.pack(fill=tk.X, pady=5)
        self.info_label = tk.Label(self.info_frame, text="Выберите источник видео и объект для трекинга", font=("Arial", 9), justify=tk.LEFT, wraplength=self.window_width-50)
        self.info_label.pack(padx=10, pady=5, anchor=tk.W)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def create_source_panel(self, parent):
        source_container = tk.Frame(parent)
        source_container.pack(fill=tk.X, padx=10, pady=10)
        button_config = {'font': ('Arial', 9), 'width': 15, 'height': 1}
        tk.Button(source_container, text="Загрузить видео файл", command=self.load_video, **button_config).pack(side=tk.LEFT, padx=5)
        tk.Button(source_container, text="Подключить камеру", command=self.connect_camera, **button_config).pack(side=tk.LEFT, padx=5)
        tk.Button(source_container, text="Отключить камеру", command=self.disconnect_camera, **button_config).pack(side=tk.LEFT, padx=5)
        camera_settings_frame = tk.Frame(source_container)
        camera_settings_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(camera_settings_frame, text="Индекс камеры:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.camera_index_var = tk.StringVar(value="0")
        camera_index_entry = tk.Entry(camera_settings_frame, textvariable=self.camera_index_var, width=5)
        camera_index_entry.pack(side=tk.LEFT, padx=5)

    def create_settings_panel(self):
        settings_container = tk.Frame(self.settings_frame)
        settings_container.pack(fill=tk.X, padx=10, pady=10)
        csrt_frame = tk.LabelFrame(settings_container, text="CSRT Параметры", font=("Arial", 9))
        csrt_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        tk.Label(csrt_frame, text="Чувствительность:", font=("Arial", 8)).pack(anchor=tk.W, pady=2)
        self.csrt_sensitivity = tk.Scale(csrt_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=120, showvalue=True)
        self.csrt_sensitivity.set(6)
        self.csrt_sensitivity.pack(fill=tk.X, padx=5, pady=2)
        kcf_frame = tk.LabelFrame(settings_container, text="KCF Параметры", font=("Arial", 9))
        kcf_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        tk.Label(kcf_frame, text="Скорость обработки:", font=("Arial", 8)).pack(anchor=tk.W, pady=2)
        self.kcf_speed = tk.Scale(kcf_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=120, showvalue=True)
        self.kcf_speed.set(8)
        self.kcf_speed.pack(fill=tk.X, padx=5, pady=2)
        mosse_frame = tk.LabelFrame(settings_container, text="MOSSE Параметры", font=("Arial", 9))
        mosse_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        tk.Label(mosse_frame, text="Скорость обучения:", font=("Arial", 8)).pack(anchor=tk.W, pady=2)
        self.mosse_learning = tk.Scale(mosse_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=120, showvalue=True)
        self.mosse_learning.set(9)
        self.mosse_learning.pack(fill=tk.X, padx=5, pady=2)
        custom_frame = tk.LabelFrame(settings_container, text="Custom CSRT Параметры", font=("Arial", 9))
        custom_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        tk.Label(custom_frame, text="Размер окна поиска:", font=("Arial", 8)).pack(anchor=tk.W, pady=2)
        self.custom_window = tk.Scale(custom_frame, from_=10, to=50, orient=tk.HORIZONTAL, length=120, showvalue=True)
        self.custom_window.set(30)
        self.custom_window.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(custom_frame, text="Скорость обучения:", font=("Arial", 8)).pack(anchor=tk.W, pady=2)
        self.custom_learning = tk.Scale(custom_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=120, showvalue=True)
        self.custom_learning.set(25)
        self.custom_learning.pack(fill=tk.X, padx=5, pady=2)

    def create_control_panel(self):
        control_subframe = tk.Frame(self.control_frame)
        control_subframe.pack()
        button_config = {'font': ('Arial', 9), 'width': 12, 'height': 1}
        tk.Button(control_subframe, text="Выбрать объект", command=self.select_object, **button_config).pack(side=tk.LEFT, padx=3)
        tk.Button(control_subframe, text="Старт трекинг", command=self.start_tracking, **button_config).pack(side=tk.LEFT, padx=3)
        tk.Button(control_subframe, text="Стоп", command=self.stop_tracking, **button_config).pack(side=tk.LEFT, padx=3)
        tk.Button(control_subframe, text="Статистика", command=self.show_statistics, **button_config).pack(side=tk.LEFT, padx=3)
        tk.Button(control_subframe, text="Сброс", command=self.reset_tracking, **button_config).pack(side=tk.LEFT, padx=3)

    def init_trackers(self):
        try:
            self.tracker_types = {
                'csrt': cv2.legacy.TrackerCSRT_create,
                'kcf': cv2.legacy.TrackerKCF_create,
                'mosse': cv2.legacy.TrackerMOSSE_create,
                'custom_csrt': CustomCSRTTracker
            }
            self.update_info("Трекеры инициализированы (3 OpenCV + 1 кастомный)")
        except AttributeError as e:
            self.update_info(f"Ошибка инициализации трекеров: {e}")
            self.tracker_types = {}

    def load_video(self):
        self.disconnect_camera()
        file_path = filedialog.askopenfilename(title="Выберите видео файл", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.camera_mode = False
            self.tracking = False
            if not self.cap.isOpened():
                self.update_info(f"Ошибка загрузки видео: {file_path}")
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            info_text = (f"Загружено видео: {file_path}\n"
                        f"Разрешение: {width}x{height}, FPS: {fps:.2f}, "
                        f"Кадров: {frame_count}, Длительность: {duration:.2f}с")
            self.update_info(info_text)
            self.auto_adjust_display_size(width, height)
            self.show_frame()

    def connect_camera(self):
        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            camera_index = 0
        self.cap = cv2.VideoCapture(camera_index)
        self.camera_mode = True
        self.tracking = False
        if not self.cap.isOpened():
            self.update_info(f"Ошибка подключения к камере с индексом {camera_index}")
            return
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        info_text = (f"Подключена камера (индекс {camera_index})\n"
                    f"Разрешение: {width}x{height}, FPS: {fps:.2f}")
        self.update_info(info_text)
        self.auto_adjust_display_size(width, height)
        self.update_camera_frame()

    def disconnect_camera(self):
        if self.cap and self.camera_mode:
            self.cap.release()
            self.cap = None
            self.camera_mode = False
            self.tracking = False
            self.update_info("Камера отключена")
            for name in ['original', 'csrt', 'kcf', 'mosse', 'custom_csrt']:
                self.video_frames[name].configure(image='')

    def update_camera_frame(self):
        if self.camera_mode and self.cap and not self.tracking:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                display_frame = self.resize_frame(frame)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frames['original'].configure(image=imgtk)
                self.video_frames['original'].image = imgtk
                for name in ['csrt', 'kcf', 'mosse', 'custom_csrt']:
                    self.video_frames[name].configure(image='')
                self.root.after(30, self.update_camera_frame)
            else:
                self.update_info("Ошибка получения кадра с камеры")

    def auto_adjust_display_size(self, original_width, original_height):
        max_width = self.display_width
        max_height = self.display_height
        scale_width = max_width / original_width
        scale_height = max_height / original_height
        scale = min(scale_width, scale_height, 1.0)
        self.display_width = int(original_width * scale)
        self.display_height = int(original_height * scale)
        self.update_info(f"Размер отображения: {self.display_width}x{self.display_height}")

    def update_info(self, message):
        self.info_label.config(text=message)
        print(message)

    def show_frame(self):
        if self.cap and not self.camera_mode and not self.tracking:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                display_frame = self.resize_frame(frame)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frames['original'].configure(image=imgtk)
                self.video_frames['original'].image = imgtk
                for name in ['csrt', 'kcf', 'mosse', 'custom_csrt']:
                    self.video_frames[name].configure(image='')

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.display_width, self.display_height))

    def select_object(self):
        if self.current_frame is not None:
            display_frame = self.resize_frame(self.current_frame)
            bbox = cv2.selectROI("Выберите объект для трекинга - выделите область и нажмите ENTER", display_frame, False)
            cv2.destroyWindow("Выберите объект для трекинга - выделите область и нажмите ENTER")
            if bbox != (0, 0, 0, 0):
                scale_x = self.current_frame.shape[1] / self.display_width
                scale_y = self.current_frame.shape[0] / self.display_height
                original_bbox = (
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y)
                )
                self.selected_bbox = original_bbox
                self.update_info(f"Выбран объект: {original_bbox}")
                self.init_all_trackers(original_bbox)
                self.draw_bbox_on_original()

    def draw_bbox_on_original(self):
        if self.current_frame is not None and self.selected_bbox is not None:
            frame = self.current_frame.copy()
            x, y, w, h = [int(v) for v in self.selected_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "Selected Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            display_frame = self.resize_frame(frame)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frames['original'].configure(image=imgtk)
            self.video_frames['original'].image = imgtk

    def init_all_trackers(self, bbox):
        self.trackers = {}
        success_count = 0
        for name, tracker_func in self.tracker_types.items():
            try:
                if name == 'custom_csrt':
                    tracker = tracker_func()
                    tracker.window_size = self.custom_window.get()
                    tracker.learning_rate = self.custom_learning.get() / 100.0
                else:
                    tracker = tracker_func()
                if self.current_frame is not None:
                    success = tracker.init(self.current_frame, bbox)
                    if success:
                        self.trackers[name] = tracker
                        success_count += 1
                        self.update_info(f"Трекер {name.upper()} инициализирован успешно")
                    else:
                        self.update_info(f"Ошибка инициализации трекера {name.upper()}")
            except Exception as e:
                self.update_info(f"Ошибка создания трекера {name.upper()}: {e}")
        if success_count > 0:
            self.update_info(f"Успешно инициализировано трекеров: {success_count}/4")
        else:
            self.update_info("Не удалось инициализировать ни один трекер")

    def start_tracking(self):
        if self.cap and self.trackers:
            self.tracking = True
            self.performance_data.clear()
            if not self.camera_mode:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_info("Трекинг запущен...")
            self.track_objects()
        else:
            self.update_info("Ошибка: видео не загружено или объект не выбран")

    def track_objects(self):
        if self.tracking and self.cap:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.camera_mode:
                    self.root.after(10, self.track_objects)
                    return
                else:
                    self.stop_tracking()
                    self.update_info("Видео завершено")
                    return
            self.current_frame = frame
            tracker_results = {}
            for name, tracker in self.trackers.items():
                try:
                    tracker_start = time.time()
                    if name == 'custom_csrt':
                        tracker.window_size = self.custom_window.get()
                        tracker.learning_rate = self.custom_learning.get() / 100.0
                    success, bbox = tracker.update(frame)
                    tracker_time = time.time() - tracker_start
                    tracker_results[name] = {
                        'success': success,
                        'bbox': bbox if success else None,
                        'time': tracker_time
                    }
                    self.collect_performance_data(name, success, bbox, tracker_time)
                except Exception as e:
                    self.update_info(f"Ошибка в трекере {name.upper()}: {e}")
                    tracker_results[name] = {'success': False, 'bbox': None, 'time': 0}
            self.display_frames(frame, tracker_results)
            processing_time = time.time() - start_time
            target_delay = max(1, int(1000 / 30 - processing_time * 1000))
            self.root.after(target_delay, self.track_objects)

    def visualize_tracking(self, frame, tracker_name, success, bbox):
        if success:
            x, y, w, h = [int(v) for v in bbox]
            colors = {
                'csrt': (255, 0, 0),
                'kcf': (0, 255, 0),
                'mosse': (0, 0, 255),
                'custom_csrt': (255, 255, 0)
            }
            color = colors.get(tracker_name, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            status = "OK" if success else "LOST"
            tracker_label = "OUR_CSRT" if tracker_name == "custom_csrt" else tracker_name.upper()
            cv2.putText(frame, f"{tracker_label} {status}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def display_frames(self, original_frame, results):
        display_orig = self.resize_frame(original_frame)
        orig_rgb = cv2.cvtColor(display_orig, cv2.COLOR_BGR2RGB)
        orig_img = Image.fromarray(orig_rgb)
        orig_tk = ImageTk.PhotoImage(image=orig_img)
        self.video_frames['original'].configure(image=orig_tk)
        self.video_frames['original'].image = orig_tk
        for name in ['csrt', 'kcf', 'mosse', 'custom_csrt']:
            display_frame = original_frame.copy()
            if name in results:
                result = results[name]
                self.visualize_tracking(display_frame, name, result['success'], result['bbox'])
            display_resized = self.resize_frame(display_frame)
            display_rgb = cv2.cvtColor(display_resized, cv2.COLOR_BGR2RGB)
            display_img = Image.fromarray(display_rgb)
            display_tk = ImageTk.PhotoImage(image=display_img)
            self.video_frames[name].configure(image=display_tk)
            self.video_frames[name].image = display_tk

    def collect_performance_data(self, tracker_name, success, bbox, processing_time):
        self.performance_data[tracker_name].append({
            'success': success,
            'bbox': bbox,
            'processing_time': processing_time,
            'timestamp': time.time()
        })

    def show_statistics(self):
        if not self.performance_data:
            self.update_info("Нет данных для статистики")
            return
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика трекинга - Сравнительный анализ")
        stats_window.geometry("800x550")
        stats_window.transient(self.root)
        stats_window.grab_set()
        metrics = {}
        for tracker_name, data in self.performance_data.items():
            if data:
                total_frames = len(data)
                success_frames = sum(1 for d in data if d['success'])
                success_rate = (success_frames / total_frames) * 100
                avg_processing_time = np.mean([d['processing_time'] for d in data])
                fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                metrics[tracker_name] = {
                    'Успешных кадров': f"{success_frames}/{total_frames}",
                    'Процент успеха': f"{success_rate:.2f}%",
                    'Среднее время обработки': f"{avg_processing_time*1000:.2f} мс",
                    'FPS': f"{fps:.2f}"
                }
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        table_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Сводная таблица")
        tree = ttk.Treeview(table_frame, columns=('Метрика', 'CSRT', 'KCF', 'MOSSE', 'OUR_CSRT'), show='headings')
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 9, 'bold'))
        style.configure("Treeview", font=('Arial', 9), rowheight=25)
        tree.heading('Метрика', text='Метрика')
        tree.heading('CSRT', text='CSRT')
        tree.heading('KCF', text='KCF')
        tree.heading('MOSSE', text='MOSSE')
        tree.heading('OUR_CSRT', text='OUR_CSRT')
        tree.column('Метрика', width=180, anchor=tk.W)
        tree.column('CSRT', width=120, anchor=tk.CENTER)
        tree.column('KCF', width=120, anchor=tk.CENTER)
        tree.column('MOSSE', width=120, anchor=tk.CENTER)
        tree.column('OUR_CSRT', width=120, anchor=tk.CENTER)
        metric_names = ['Успешных кадров', 'Процент успеха', 'Среднее время обработки', 'FPS']
        for metric in metric_names:
            row = [metric]
            for tracker in ['csrt', 'kcf', 'mosse', 'custom_csrt']:
                if tracker in metrics and metric in metrics[tracker]:
                    row.append(metrics[tracker][metric])
                else:
                    row.append('N/A')
            tree.insert('', tk.END, values=row)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        conclusions_frame = ttk.Frame(notebook)
        notebook.add(conclusions_frame, text="Анализ и выводы")
        conclusions_text = tk.Text(conclusions_frame, wrap=tk.WORD, font=('Arial', 10), padx=10, pady=10, height=15)
        scrollbar_text = ttk.Scrollbar(conclusions_frame, orient=tk.VERTICAL, command=conclusions_text.yview)
        conclusions_text.configure(yscrollcommand=scrollbar_text.set)
        scrollbar_text.pack(side=tk.RIGHT, fill=tk.Y)
        conclusions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_text = self.generate_analysis_text(metrics)
        conclusions_text.insert(tk.END, analysis_text)
        conclusions_text.config(state=tk.DISABLED)
        algorithms_frame = ttk.Frame(notebook)
        notebook.add(algorithms_frame, text="Описание алгоритмов")
        algorithms_text = tk.Text(algorithms_frame, wrap=tk.WORD, font=('Arial', 10), padx=10, pady=10, height=15)
        scrollbar_algo = ttk.Scrollbar(algorithms_frame, orient=tk.VERTICAL, command=algorithms_text.yview)
        algorithms_text.configure(yscrollcommand=scrollbar_algo.set)
        scrollbar_algo.pack(side=tk.RIGHT, fill=tk.Y)
        algorithms_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        algorithms_description = self.generate_algorithms_description()
        algorithms_text.insert(tk.END, algorithms_description)
        algorithms_text.config(state=tk.DISABLED)
        button_frame = tk.Frame(stats_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Сохранить статистику", command=self.save_statistics, font=('Arial', 9), width=18, height=1).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Экспорт в CSV", command=self.export_to_csv, font=('Arial', 9), width=15, height=1).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Закрыть", command=stats_window.destroy, font=('Arial', 9), width=15, height=1).pack(side=tk.LEFT, padx=5)

    def generate_analysis_text(self, metrics):
        text = "СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ТРЕКИНГА\n\n"
        text += "="*60 + "\n\n"
        if all(tracker in metrics for tracker in ['csrt', 'kcf', 'mosse', 'custom_csrt']):
            csrt_acc = float(metrics['csrt']['Процент успеха'].replace('%', ''))
            kcf_acc = float(metrics['kcf']['Процент успеха'].replace('%', ''))
            mosse_acc = float(metrics['mosse']['Процент успеха'].replace('%', ''))
            custom_acc = float(metrics['custom_csrt']['Процент успеха'].replace('%', ''))
            text += "1. ТОЧНОСТЬ ТРЕКИНГА:\n"
            accuracies = [('CSRT', csrt_acc), ('KCF', kcf_acc), ('MOSSE', mosse_acc), ('OUR_CSRT', custom_acc)]
            accuracies.sort(key=lambda x: x[1], reverse=True)
            for i, (name, acc) in enumerate(accuracies):
                if i == 0:
                    text += f"   • {name}: {acc:.1f}% - наивысшая точность\n"
                else:
                    text += f"   • {name}: {acc:.1f}%\n"
            text += "\n"
            csrt_fps = float(metrics['csrt']['FPS'])
            kcf_fps = float(metrics['kcf']['FPS'])
            mosse_fps = float(metrics['mosse']['FPS'])
            custom_fps = float(metrics['custom_csrt']['FPS'])
            text += "2. СКОРОСТЬ ОБРАБОТКИ:\n"
            speeds = [('MOSSE', mosse_fps), ('KCF', kcf_fps), ('OUR_CSRT', custom_fps), ('CSRT', csrt_fps)]
            speeds.sort(key=lambda x: x[1], reverse=True)
            for i, (name, fps) in enumerate(speeds):
                if i == 0:
                    text += f"   • {name}: {fps:.1f} FPS - наивысшая скорость\n"
                else:
                    text += f"   • {name}: {fps:.1f} FPS\n"
            text += "\n"
            text += "3. РЕКОМЕНДАЦИИ ПО ПРИМЕНЕНИЮ:\n"
            text += "   • CSRT: Для задач, требующих максимальной точности\n"
            text += "   • KCF: Для сбалансированных real-time приложений\n"
            text += "   • MOSSE: Для high-speed трекинга с простыми объектами\n"
            text += "   • OUR_CSRT: Для понимания принципов работы трекеров\n\n"
            text += "4. АНАЛИЗ НАШЕЙ РЕАЛИЗАЦИИ CSRT:\n"
            if custom_acc > 70:
                text += "   • Отличная работа! Наша реализация показывает хорошие результаты\n"
            elif custom_acc > 50:
                text += "   • Хорошая работа! Есть потенциал для улучшения\n"
            else:
                text += "   • Требуется доработка алгоритма для улучшения точности\n"
            text += "\nВЫВОД: Выбор метода зависит от требований к точности и скорости.\n"
        return text

    def generate_algorithms_description(self):
        text = "ОПИСАНИЕ АЛГОРИТМОВ ТРЕКИНГА\n\n"
        text += "="*50 + "\n\n"
        text += "1. CSRT (Channel and Spatial Reliability Tracker):\n"
        text += "   - Использует пространственную и канальную надежность\n"
        text += "   - Высокая точность, но низкая скорость\n"
        text += "   - Хорошо работает с изменяющейся формой объекта\n\n"
        text += "2. KCF (Kernelized Correlation Filters):\n"
        text += "   - Быстрый алгоритм на основе ядерных фильтров\n"
        text += "   - Баланс между точностью и скоростью\n"
        text += "   - Эффективен для real-time приложений\n\n"
        text += "3. MOSSE (Minimum Output Sum of Squared Error):\n"
        text += "   - Очень быстрый адаптивный трекер\n"
        text += "   - Высокая скорость, но меньшая точность\n"
        text += "   - Идеален для систем с ограниченными ресурсами\n\n"
        text += "4. OUR CSRT (Наша реализация):\n"
        text += "   - Упрощенная версия CSRT трекера\n"
        text += "   - Использует цветовые гистограммы и корреляцию\n"
        text += "   - Понимание принципов работы трекеров\n"
        text += "   - Параметры:\n"
        text += "     * Размер окна поиска: область вокруг объекта для поиска\n"
        text += "     * Скорость обучения: как быстро обновляется модель\n\n"
        text += "МАТЕМАТИЧЕСКАЯ МОДЕЛЬ OUR CSRT:\n"
        text += "   - Гистограммы в HSV и LAB пространствах\n"
        text += "   - Корреляция шаблонов для поиска объекта\n"
        text += "   - Взвешенная оценка: 0.6 * корреляция + 0.4 * сходство гистограмм\n"
        text += "   - Адаптивное обновление модели: new = (1-α)*old + α*current\n"
        return text

    def save_statistics(self):
        if self.performance_data:
            filename = f"tracking_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
            stats = {
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'video_source': 'camera' if self.camera_mode else 'file',
                'frames_processed': len(self.performance_data.get('csrt', [])),
                'trackers': {}
            }
            for tracker_name, data in self.performance_data.items():
                if data:
                    total_frames = len(data)
                    success_frames = sum(1 for d in data if d['success'])
                    success_rate = (success_frames / total_frames) * 100
                    avg_processing_time = np.mean([d['processing_time'] for d in data])
                    fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    display_name = 'OUR_CSRT' if tracker_name == 'custom_csrt' else tracker_name.upper()
                    stats['trackers'][display_name] = {
                        'total_frames': total_frames,
                        'success_frames': success_frames,
                        'success_rate': success_rate,
                        'average_processing_time_ms': avg_processing_time * 1000,
                        'fps': fps
                    }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.update_info(f"Статистика сохранена в {filename}")

    def export_to_csv(self):
        if self.performance_data:
            filename = f"tracking_analysis_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Метод трекинга,Успешных кадров,Процент успеха,Среднее время (мс),FPS\n")
                    for tracker_name, data in self.performance_data.items():
                        if data:
                            total_frames = len(data)
                            success_frames = sum(1 for d in data if d['success'])
                            success_rate = (success_frames / total_frames) * 100
                            avg_processing_time = np.mean([d['processing_time'] for d in data])
                            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                            display_name = 'OUR_CSRT' if tracker_name == 'custom_csrt' else tracker_name.upper()
                            f.write(f"{display_name},{success_frames}/{total_frames},{success_rate:.2f}%,{avg_processing_time*1000:.2f},{fps:.2f}\n")
                self.update_info(f"Данные экспортированы в {filename}")
            except Exception as e:
                self.update_info(f"Ошибка экспорта: {e}")

    def stop_tracking(self):
        self.tracking = False
        self.update_info("Трекинг остановлен")
        if self.camera_mode:
            self.update_camera_frame()

    def reset_tracking(self):
        self.stop_tracking()
        self.selected_bbox = None
        self.trackers = {}
        self.performance_data.clear()
        if self.cap and not self.camera_mode:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.show_frame()
        elif self.camera_mode:
            self.update_camera_frame()
        self.update_info("Трекинг сброшен. Выберите новый объект для трекинга.")

def main():
    print(f"OpenCV version: {cv2.__version__}")
    legacy_trackers = ['TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMOSSE_create']
    for tracker in legacy_trackers:
        if hasattr(cv2.legacy, tracker):
            print(f"[OK] {tracker} доступен в cv2.legacy")
        else:
            print(f"[FAIL] {tracker} недоступен в cv2.legacy")
    root = tk.Tk()
    app = ObjectTrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
