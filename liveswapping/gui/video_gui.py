# -*- coding: utf-8 -*-
"""Красивый Qt-GUI для видео обработки на основе скомпилированного .ui файла."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import List

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QWidget, QMessageBox, QFileDialog
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont

# Импорт скомпилированного UI
from .video_gui_ui import Ui_VideoGUI

# Локализация
import liveswapping.utils.localisation as loc

from liveswapping.run import run as cli_run
from liveswapping.ai_models.download_models import MODELS, ensure_model


class VideoWorker(threading.Thread):
    def __init__(self, args: List[str]):
        super().__init__(daemon=True)
        self.args = args
        self.exc: Exception | None = None
        self._stop_event = threading.Event()
        self._process = None

    def run(self):
        try:
            import subprocess
            import sys
            
            print(f"[DEBUG] VideoWorker.run() started with args: {self.args}")
            
            # Запускаем процесс как подпроцесс для возможности его остановки
            # Используем корневой run.py вместо -m liveswapping.run для избежания проблем с Python path
            run_script = Path(__file__).parent.parent.parent / "run.py"
            cmd = [sys.executable, str(run_script)] + self.args
            print(f"[DEBUG] Executing command: {' '.join(cmd)}")
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            
            print(f"[DEBUG] Process started with PID: {self._process.pid}")
            
            # Ждем завершения процесса или сигнала остановки
            while self._process.poll() is None:
                if self._stop_event.wait(timeout=0.1):
                    # Получили сигнал остановки
                    print("[DEBUG] Stop signal received, terminating process")
                    self._terminate_process()
                    break
            
            # Проверяем код возврата
            returncode = self._process.returncode
            print(f"[DEBUG] Process finished with return code: {returncode}")
            
            if self._process and returncode not in [0, None]:
                stdout_output = self._process.stdout.read() if self._process.stdout else ""
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                print(f"[DEBUG] Process stdout: {stdout_output}")
                print(f"[DEBUG] Process stderr: {stderr_output}")
                if stderr_output and not self._stop_event.is_set():
                    raise Exception(f"Process failed with code {returncode}: {stderr_output}")
                    
        except Exception as e:
            print(f"[DEBUG] Exception in VideoWorker.run(): {e}")
            if not self._stop_event.is_set():  # Не показываем ошибку если мы сами остановили
                self.exc = e

    def _terminate_process(self):
        """Корректное завершение процесса."""
        if not self._process:
            return
            
        try:
            import sys
            if sys.platform == "win32":
                # На Windows используем taskkill для корректного завершения
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self._process.pid)], 
                             check=False, capture_output=True)
            else:
                # На Unix системах
                self._process.terminate()
                
            # Ждем завершения
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Если не завершился, принудительно убиваем
                if sys.platform == "win32":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self._process.pid)], 
                                 check=False, capture_output=True)
                else:
                    self._process.kill()
                    self._process.wait(timeout=2)
        except Exception:
            # В крайнем случае пытаемся kill
            try:
                self._process.kill()
            except:
                pass

    def stop(self):
        """Остановка выполнения процесса."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._terminate_process()

    def is_running(self):
        """Проверка, выполняется ли процесс."""
        return self._process is not None and self._process.poll() is None


class VideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: VideoWorker | None = None
        self._current_mode: int = 1  # 0 = Image to Image, 1 = Image to Video (по умолчанию)
        
        # Загрузка UI
        self.ui = Ui_VideoGUI()
        self.ui.setupUi(self)
        
        self._setup_font()
        self._setup_theme()
        
        # Сначала инициализируем язык по умолчанию
        self._init_language()
        
        self._setup_data()
        self._connect_signals()
        self._model_selected()
        
        # Применяем локализацию при запуске
        self._update_ui_texts()
        
    def _init_language(self):
        """Инициализация языка по умолчанию."""
        # Устанавливаем английский язык по умолчанию
        loc.set_language("en")
        
        # Настраиваем комбобокс языков без вызова сигналов
        self.ui.languageCombo.blockSignals(True)
        self.ui.languageCombo.clear()
        self.ui.languageCombo.addItem("English")
        self.ui.languageCombo.addItem("Русский")
        self.ui.languageCombo.setCurrentIndex(0)  # English по умолчанию
        self.ui.languageCombo.blockSignals(False)

    def _setup_font(self):
        """Настройка простого системного шрифта."""
        # Простой системный шрифт
        app_font = QFont()
        app_font.setPointSize(10)
        app_font.setWeight(QFont.Weight.Normal)
        
        # Устанавливаем для основного виджета
        self.setFont(app_font)

    def _setup_theme(self):
        """Настройка киберпанк темы в сине-аквамариновых тонах."""
        style = """
        /* =================================================================
           КИБЕРПАНК ТЕМА (Сине-аквамариновые тона)
           ================================================================= */
        
        QWidget {
            background-color: #0a0e1a;
            color: #e0f7ff;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 10px;
            selection-background-color: #00bcd4;
            selection-color: #000d1a;
        }
        
        /* =================================================================
           ГРУППЫ (GroupBox) - Киберпанк стиль
           ================================================================= */
        
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            border: 2px solid #1e3a5f;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            background-color: #0d1b2a;
            color: #00e5ff;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #00e5ff;
            background-color: #0d1b2a;
            font-weight: bold;
        }
        
        /* =================================================================
           ПОЛЯ ВВОДА (LineEdit) - Киберпанк стиль
           ================================================================= */
        
        QLineEdit {
            border: 2px solid #1a4480;
            border-radius: 6px;
            padding: 10px 14px;
            background-color: #0f1419;
            color: #e0f7ff;
            font-size: 10px;
        }
        
        QLineEdit:focus {
            border-color: #00bcd4;
            background-color: #162026;
        }
        
        QLineEdit:read-only {
            background-color: #0a0f14;
            color: #7fb3d3;
            border-color: #284666;
        }
        
        QLineEdit::placeholder {
            color: #4a90a4;
            font-style: italic;
        }
        
        /* =================================================================
           КНОПКИ (QPushButton) - Киберпанк стиль
           ================================================================= */
        
        QPushButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #1565c0, stop: 1 #0d47a1);
            color: #e0f7ff;
            border: 2px solid #1976d2;
            border-radius: 6px;
            padding: 10px 16px;
            font-weight: 600;
            font-size: 10px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #1976d2, stop: 1 #1565c0);
            border-color: #00bcd4;
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #0d47a1, stop: 1 #01579b);
            border-color: #0097a7;
        }
        
        QPushButton:disabled {
            background-color: #263238;
            color: #455a64;
            border-color: #37474f;
        }
        
        /* Кнопка Start (неоново-зеленая киберпанк) */
        QPushButton#startButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #00e676, stop: 1 #00c853);
            color: #000d1a;
            border: 2px solid #00e676;
            font-size: 12px;
            font-weight: bold;
        }
        
        QPushButton#startButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #1de9b6, stop: 1 #00e676);
            border-color: #1de9b6;
        }
        
        QPushButton#startButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #00c853, stop: 1 #00a846);
        }
        
        /* Кнопка Stop (неоново-красная киберпанк) */
        QPushButton#stopButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #ff1744, stop: 1 #d50000);
            color: #ffffff;
            border: 2px solid #ff1744;
            font-size: 12px;
            font-weight: bold;
        }
        
        QPushButton#stopButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #ff5983, stop: 1 #ff1744);
            border-color: #ff5983;
        }
        
        QPushButton#stopButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #d50000, stop: 1 #b71c1c);
        }
        
        QPushButton#stopButton:disabled {
            background-color: #263238;
            color: #455a64;
            border-color: #37474f;
        }
        
        /* =================================================================
           ВЫПАДАЮЩИЕ СПИСКИ (QComboBox) - Киберпанк стиль
           ================================================================= */
        
        QComboBox {
            border: 2px solid #1a4480;
            border-radius: 6px;
            padding: 8px 30px 8px 12px;
            background-color: #0f1419;
            color: #e0f7ff;
            min-width: 140px;
        }
        
        QComboBox:focus {
            border-color: #00bcd4;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 25px;
            background-color: #1565c0;
            border-radius: 0 4px 4px 0;
        }
        
        QComboBox::down-arrow {
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid #e0f7ff;
            margin-right: 8px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #0d1b2a;
            color: #e0f7ff;
            border: 2px solid #00bcd4;
            border-radius: 6px;
            selection-background-color: #1565c0;
        }
        
        QComboBox QAbstractItemView::item {
            padding: 10px 14px;
            border: none;
        }
        
        QComboBox QAbstractItemView::item:selected {
            background-color: #1976d2;
            color: #ffffff;
        }
        
        QComboBox QAbstractItemView::item:hover {
            background-color: #1565c0;
            color: #00e5ff;
        }
        
        /* =================================================================
           ЧИСЛОВЫЕ ПОЛЯ (QSpinBox) - Киберпанк стиль
           ================================================================= */
        
        QSpinBox {
            border: 2px solid #1a4480;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #0f1419;
            color: #e0f7ff;
            min-width: 80px;
        }
        
        QSpinBox:focus {
            border-color: #00bcd4;
        }
        
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #1565c0;
            border: none;
            width: 20px;
        }
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background-color: #1976d2;
        }
        
        QSpinBox::up-arrow {
            width: 0;
            height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 6px solid #e0f7ff;
        }
        
        QSpinBox::down-arrow {
            width: 0;
            height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #e0f7ff;
        }
        
        /* =================================================================
           ЧЕКБОКСЫ (QCheckBox) - Киберпанк стиль
           ================================================================= */
        
        QCheckBox {
            color: #e0f7ff;
            font-size: 11px;
            font-weight: 500;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #1a4480;
            border-radius: 4px;
            background-color: #0f1419;
        }
        
        QCheckBox::indicator:hover {
            border-color: #00bcd4;
            background-color: #162026;
        }
        
        QCheckBox::indicator:checked {
            background-color: #00e676;
            border-color: #00e676;
        }
        
        QCheckBox::indicator:checked:hover {
            background-color: #1de9b6;
            border-color: #1de9b6;
        }
        
        /* =================================================================
           МЕТКИ (QLabel) - Киберпанк стиль  
           ================================================================= */
        
        QLabel {
            color: #e0f7ff;
            font-weight: 500;
        }
        
        /* Информационные лейблы с меньшей яркостью */
        QLabel[objectName*="InfoLabel"] {
            color: #4a90a4;
            font-style: italic;
            font-size: 9px;
        }
        
        /* =================================================================
           ПРОГРЕСС-БАР (QProgressBar) - Киберпанк стиль
           ================================================================= */
        
        QProgressBar {
            border: 2px solid #1a4480;
            border-radius: 6px;
            background-color: #0f1419;
            color: #e0f7ff;
            text-align: center;
            font-weight: bold;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 #00e676, stop: 0.5 #00bcd4, stop: 1 #1976d2);
            border-radius: 4px;
        }
        
        /* =================================================================
           СКРОЛЛ ОБЛАСТЬ
           ================================================================= */
        
        QScrollArea {
            border: none;
            background-color: #0a0e1a;
        }
        
        QScrollBar:vertical {
            background-color: #0d1b2a;
            width: 12px;
            border-radius: 6px;
            border: 1px solid #1a4480;
        }
        
        QScrollBar::handle:vertical {
            background-color: #1565c0;
            border-radius: 5px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #1976d2;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        """
        
        self.setStyleSheet(style)

    def _setup_data(self):
        """Настройка данных для комбобоксов."""
        # Языки уже настроены в _init_language()
        
        # Провайдеры для модели - заполняем программно с автоопределением
        self.ui.modelProviderCombo.clear()
        providers = self._detect_available_providers()
        for provider_id, provider_name in providers:
            self.ui.modelProviderCombo.addItem(provider_name, provider_id)
        
        # Провайдеры для upscaler - заполняем программно с автоопределением  
        self.ui.upscalerProviderCombo.clear()
        for provider_id, provider_name in providers:
            self.ui.upscalerProviderCombo.addItem(provider_name, provider_id)
        
        # Устанавливаем оптимальный провайдер по умолчанию для обоих
        optimal_provider = self._get_optimal_provider(providers)
        for combo in [self.ui.modelProviderCombo, self.ui.upscalerProviderCombo]:
            for i in range(combo.count()):
                if combo.itemData(i) == optimal_provider:
                    combo.setCurrentIndex(i)
                    break
        
        # Модели уже настроены в .ui файле
        
        # Апскейлеры уже настроены в .ui файле
        
        # Инициализация состояния чекбокса апскейлера
        if hasattr(self.ui, 'upscalerCheckBox'):
            self.ui.upscalerCheckBox.setChecked(False)  # По умолчанию выключен
            # Скрываем настройки апскейлера если чекбокс выключен
            if hasattr(self.ui, 'upscalerOptionsFrame'):
                self.ui.upscalerOptionsFrame.setVisible(False)

    def _connect_signals(self):
        """Подключение сигналов."""
        self.ui.sourceBrowseButton.clicked.connect(self._browse_source)
        self.ui.targetBrowseButton.clicked.connect(self._browse_target)
        self.ui.modelBrowseButton.clicked.connect(self._browse_model)
        self.ui.modeCombo.currentIndexChanged.connect(self._mode_changed)
        self.ui.modelCombo.currentIndexChanged.connect(self._model_selected)
        # Подключаем сигнал upscaler checkbox если он есть в UI
        if hasattr(self.ui, 'upscalerCheckBox'):
            self.ui.upscalerCheckBox.toggled.connect(self._upscaler_toggled)
        self.ui.startButton.clicked.connect(self._start)
        self.ui.stopButton.clicked.connect(self._stop)
        self.ui.languageCombo.currentIndexChanged.connect(self._change_language)

    # ------------------------------------------------------------------
    # Provider detection and selection (simplified for basic UI)
    # ------------------------------------------------------------------
    def _detect_available_providers(self):
        """Определение доступных провайдеров на системе."""
        providers = []
        
        # CPU - всегда доступен
        providers.append(("cpu", loc.get("provider_cpu")))
        
        try:
            # Проверяем CUDA
            import torch
            if torch.cuda.is_available():
                providers.append(("cuda", loc.get("provider_cuda")))
        except ImportError:
            pass
            
        return providers
    
    def _get_optimal_provider(self, providers):
        """Определение оптимального провайдера из доступных."""
        provider_priority = ["cuda", "cpu"]
        
        available_ids = [p[0] for p in providers]
        
        for preferred in provider_priority:
            if preferred in available_ids:
                return preferred
                
        return "cpu"  # Fallback
    
    def _model_provider_selected(self):
        """Обработка смены провайдера для модели."""
        current_provider = self.ui.modelProviderCombo.currentData()
        # Тихо обновляем, без логов при каждом изменении
        pass
    
    def _upscaler_provider_selected(self):
        """Обработка смены провайдера для upscaler."""
        current_provider = self.ui.upscalerProviderCombo.currentData()
        # Тихо обновляем, без логов при каждом изменении
        pass
    
    def _upscaler_toggled(self, checked):
        """Обработка переключения чекбокса апскейлера."""
        # Показываем/скрываем настройки апскейлера в зависимости от состояния чекбокса
        if hasattr(self.ui, 'upscalerOptionsFrame'):
            self.ui.upscalerOptionsFrame.setVisible(checked)

    # ------------------------------------------------------------------
    # File browsers
    # ------------------------------------------------------------------
    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            loc.get("dialog_select_source"), 
            "", 
            loc.get("filter_images")
        )
        if path:
            self.ui.sourceEdit.setText(path)

    def _browse_target(self):
        """Выбор целевого видео или изображения в зависимости от режима."""
        current_mode = self.ui.modeCombo.currentIndex()
        
        if current_mode == 0:  # Image to Image
            path, _ = QFileDialog.getOpenFileName(
                self, 
                loc.get("dialog_select_target_image"), 
                "", 
                loc.get("filter_images")
            )
        else:  # Image to Video
            path, _ = QFileDialog.getOpenFileName(
                self, 
                loc.get("dialog_select_target"), 
                "", 
                loc.get("filter_videos")
            )
        
        if path:
            self.ui.targetEdit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            loc.get("dialog_select_model"), 
            "models", 
            loc.get("filter_models")
        )
        if path:
            self.ui.modelPathEdit.setText(path)

    # ------------------------------------------------------------------
    # Model selection handler
    # ------------------------------------------------------------------
    def _model_selected(self):
        current_text = self.ui.modelCombo.currentText()
        
        if "DFM Model" in current_text:
            # DFM - пользователь должен выбрать файл
            self.ui.modelPathEdit.clear()
            self.ui.modelPathEdit.setPlaceholderText(loc.get("select_dfm_file"))
            self.ui.modelBrowseButton.setEnabled(True)
            self.ui.modelPathEdit.setReadOnly(False)
            return
            
        # Встроенные модели - автозагрузка
        self.ui.modelBrowseButton.setEnabled(False)
        self.ui.modelPathEdit.setReadOnly(True)
        
        try:
            # Определяем тип модели по тексту
            if "reswapper128" in current_text:
                model_name = "reswapper128"
            elif "reswapper256" in current_text:
                model_name = "reswapper256"
            elif "inswapper128" in current_text:
                model_name = "inswapper128"
            else:
                self.ui.modelPathEdit.clear()
                self.ui.modelPathEdit.setPlaceholderText(loc.get("unknown_model_type"))
                return
            
            # Показываем статус загрузки
            self.ui.modelPathEdit.setText(loc.get("downloading_model"))
            self.ui.modelPathEdit.setPlaceholderText("")
            QApplication.processEvents()  # Обновляем UI
                
            path = ensure_model(model_name)
            self.ui.modelPathEdit.setText(str(path))
            self.ui.modelPathEdit.setPlaceholderText("")
        except Exception as e:
            QMessageBox.critical(self, loc.get("model_error"), f"{loc.get('failed_to_ensure_model')} {str(e)}")
            self.ui.modelPathEdit.clear()
            self.ui.modelPathEdit.setPlaceholderText(loc.get("model_error"))

    # ------------------------------------------------------------------
    # Mode selection handler
    # ------------------------------------------------------------------
    def _mode_changed(self):
        """Обработка смены режима (Image to Image/Image to Video)."""
        current_mode = self.ui.modeCombo.currentIndex()
        
        if current_mode == 0:  # Image to Image
            # Настройки для режима Image to Image
            self.ui.faceSwapGroup.setVisible(True)  # Модель нужна и для фото
            self.ui.upscalerGroup.setVisible(True)  # Апскейлер тоже полезен для изображений
            self.ui.settingsGroup.setVisible(True)
            self.ui.mouthCheckBox.setVisible(False)  # Сохранение рта только для видео
            
            # Меняем лейблы
            self.ui.targetLabel.setText(loc.get("target_image"))
            self.ui.targetEdit.setPlaceholderText(loc.get("placeholder_target_image"))
            
        elif current_mode == 1:  # Image to Video
            # Показываем все группы для видео
            self.ui.faceSwapGroup.setVisible(True)
            self.ui.upscalerGroup.setVisible(True) 
            self.ui.settingsGroup.setVisible(True)
            self.ui.mouthCheckBox.setVisible(True)
            
            # Меняем лейблы обратно
            self.ui.targetLabel.setText(loc.get("target_video"))
            self.ui.targetEdit.setPlaceholderText(loc.get("placeholder_target"))
        
        # Обновляем тексты кнопок
        self._update_ui_texts()

    # ------------------------------------------------------------------
    # Processing control
    # ------------------------------------------------------------------
    def _start(self):
        if self._worker:
            QMessageBox.warning(self, loc.get("already_running"), loc.get("processing_in_progress"))
            return
            
        # Validate inputs
        src_path = Path(self.ui.sourceEdit.text())
        tgt_path = Path(self.ui.targetEdit.text())
        model_path = Path(self.ui.modelPathEdit.text())
        current_mode = self.ui.modeCombo.currentIndex()
        self._current_mode = current_mode  # Сохраняем для использования в _on_worker_finished
        
        # Отладочная информация
        print(f"[DEBUG] Source path: {src_path}")
        print(f"[DEBUG] Target path: {tgt_path}")
        print(f"[DEBUG] Model path: {model_path}")
        print(f"[DEBUG] Current mode: {current_mode}")
        
        if not src_path.exists():
            QMessageBox.warning(self, loc.get("missing_source"), loc.get("select_valid_source"))
            return
        if not tgt_path.exists():
            if current_mode == 0:  # Image to Image
                QMessageBox.warning(self, loc.get("missing_target"), loc.get("select_valid_target_image"))
            else:  # Image to Video
                QMessageBox.warning(self, loc.get("missing_target"), loc.get("select_valid_target"))
            return
        if not model_path.exists():
            QMessageBox.warning(self, loc.get("missing_model"), loc.get("select_valid_model"))
            return

        # Compose arguments based on mode
        if current_mode == 0:  # Image to Image
            args = [
                "image",
                "--source", str(src_path),
                "--target", str(tgt_path),
                "--modelPath", str(model_path),
                "--resolution", str(self.ui.resolutionSpinBox.value()),
            ]
        else:  # Image to Video
            args = [
                "video",
                "--source", str(src_path),
                "--target_video", str(tgt_path),
                "--modelPath", str(model_path),
                "--resolution", str(self.ui.resolutionSpinBox.value()),
            ]
        
        # Add model provider argument
        model_provider = self.ui.modelProviderCombo.currentData()
        if model_provider and model_provider != "cpu":
            args.extend(["--model_provider", model_provider])
        
        # Add upscaler argument (для обоих режимов) - только если включен чекбокс
        upscaler_enabled = False  # По умолчанию выключен
        if hasattr(self.ui, 'upscalerCheckBox'):
            upscaler_enabled = self.ui.upscalerCheckBox.isChecked()
        
        if upscaler_enabled:
            upscaler_text = self.ui.upscalerCombo.currentText()
            if "GFPGAN" in upscaler_text:
                args.extend(["--bg_upsampler", "gfpgan"])
                # Add upscaler provider argument
                upscaler_provider = self.ui.upscalerProviderCombo.currentData()
                if upscaler_provider and upscaler_provider != "cpu":
                    args.extend(["--upscaler_provider", upscaler_provider])
            elif "RealESRGAN" in upscaler_text:
                args.extend(["--bg_upsampler", "realesrgan"])
                # Add upscaler provider argument
                upscaler_provider = self.ui.upscalerProviderCombo.currentData()
                if upscaler_provider and upscaler_provider != "cpu":
                    args.extend(["--upscaler_provider", upscaler_provider])
        # Если апскейлер выключен или "None" - не добавляем аргумент
        
        # Mouth mask только для видео
        if current_mode == 1 and self.ui.mouthCheckBox.isChecked():
            args.append("--mouth_mask")

        # Отладочная информация
        print(f"[DEBUG] Final args: {args}")
        print(f"[DEBUG] Starting VideoWorker...")

        # Start processing
        self._worker = VideoWorker(args)
        self._worker.start()
        
        print(f"[DEBUG] VideoWorker started")
        
        # Update UI
        self.ui.startButton.setEnabled(False)
        self.ui.stopButton.setEnabled(True)
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 0)  # Indeterminate progress
        if current_mode == 0:  # Image to Image
            self.ui.progressBar.setFormat(loc.get("processing_image"))
        else:  # Image to Video
            self.ui.progressBar.setFormat(loc.get("processing_video"))
        
        # Monitor worker
        threading.Thread(target=self._monitor_worker, daemon=True).start()

    def _stop(self):
        if self._worker and self._worker.is_alive():
            # Останавливаем процесс
            self._worker.stop()
            
            # Показываем уведомление об остановке
            QMessageBox.information(
                self, 
                loc.get("stop_processing"), 
                loc.get("process_stopped")
            )
            
            # Сбрасываем UI в исходное состояние
            self._reset_ui()
        else:
            self._reset_ui()

    def _monitor_worker(self):
        """Monitors the worker thread and updates UI when done."""
        if not self._worker:
            print("[DEBUG] _monitor_worker: No worker found")
            return
            
        print("[DEBUG] _monitor_worker: Waiting for worker to finish...")
        self._worker.join()
        print("[DEBUG] _monitor_worker: Worker finished, scheduling UI update")
        
        # Update UI on main thread
        QTimer.singleShot(0, self._on_worker_finished)

    def _on_worker_finished(self):
        """Called when worker finishes - updates UI."""
        print("[DEBUG] _on_worker_finished: Called")
        
        if self._worker and self._worker.exc:
            print(f"[DEBUG] Worker exception: {self._worker.exc}")
            QMessageBox.critical(
                self, 
                loc.get("processing_error"), 
                f"{loc.get('error_occurred')}\n\n{str(self._worker.exc)}"
            )
        elif self._worker and self._worker._stop_event.is_set():
            # Процесс был остановлен пользователем - не показываем сообщение
            print("[DEBUG] Process was stopped by user")
            pass
        else:
            # Процесс завершился успешно
            print("[DEBUG] Process completed successfully")
            if self._current_mode == 0:  # Image to Image
                success_message = loc.get("image_completed")
            else:  # Image to Video
                success_message = loc.get("video_completed")
            
            QMessageBox.information(
                self, 
                loc.get("success"), 
                success_message
            )
        
        self._reset_ui()

    def _reset_ui(self):
        """Resets UI to initial state."""
        self._worker = None
        self.ui.startButton.setEnabled(True)
        self.ui.stopButton.setEnabled(False)
        self.ui.progressBar.setVisible(False)

    # ------------------------------------------------------------------
    # Language change handler
    # ------------------------------------------------------------------
    def _change_language(self):
        lang = "ru" if self.ui.languageCombo.currentIndex() == 1 else "en"
        loc.set_language(lang)
        
        # Обновляем UI тексты сразу
        self._update_ui_texts()
    
    def _update_ui_texts(self):
        """Обновление всех текстов в UI согласно текущему языку."""
        # Обновляем заголовок окна
        self.setWindowTitle(loc.get("title_video"))
        
        # Обновляем группы и лейблы
        self.ui.filesGroup.setTitle(loc.get("input_files"))
        self.ui.faceSwapGroup.setTitle(loc.get("face_swap_model"))
        self.ui.upscalerGroup.setTitle(loc.get("face_enhancement"))
        self.ui.settingsGroup.setTitle(loc.get("processing_settings"))
        
        # Обновляем лейблы полей
        self.ui.sourceLabel.setText(loc.get("source_image"))
        
        # Лейбл цели зависит от режима
        current_mode = self.ui.modeCombo.currentIndex()
        if current_mode == 0:  # Image to Image
            self.ui.targetLabel.setText(loc.get("target_image"))
            self.ui.targetEdit.setPlaceholderText(loc.get("placeholder_target_image"))
        else:  # Image to Video
            self.ui.targetLabel.setText(loc.get("target_video"))
            self.ui.targetEdit.setPlaceholderText(loc.get("placeholder_target"))
        
        self.ui.modelLabel.setText(loc.get("ai_model"))
        self.ui.modelPathLabel.setText(loc.get("model_path"))
        self.ui.modelProviderLabel.setText(loc.get("provider"))
        self.ui.upscalerLabel.setText(loc.get("upscaler"))
        self.ui.upscalerProviderLabel.setText(loc.get("provider"))
        self.ui.resolutionLabel.setText(loc.get("resolution"))
        
        # Обновляем кнопки
        self.ui.sourceBrowseButton.setText(loc.get("browse"))
        self.ui.targetBrowseButton.setText(loc.get("browse"))
        self.ui.modelBrowseButton.setText(loc.get("browse"))
        self.ui.startButton.setText(loc.get("start_processing"))
        self.ui.stopButton.setText(loc.get("stop"))
        
        # Обновляем чекбоксы
        self.ui.mouthCheckBox.setText(loc.get("retain_target_mouth"))
        if hasattr(self.ui, 'upscalerCheckBox'):
            self.ui.upscalerCheckBox.setText(loc.get("enable_face_enhancement"))
        
        # Обновляем плейсхолдеры
        self.ui.sourceEdit.setPlaceholderText(loc.get("placeholder_source"))
        self.ui.modelPathEdit.setPlaceholderText(loc.get("placeholder_model"))
        
        # Обновляем провайдеров БЕЗ потери текущего состояния
        self._update_providers_texts()
    
    def _update_providers_texts(self):
        """Обновление текстов провайдеров с сохранением текущего выбора."""
        # Сохраняем текущие выбранные провайдеры
        current_model_provider = self.ui.modelProviderCombo.currentData()
        current_upscaler_provider = self.ui.upscalerProviderCombo.currentData()
        
        # Получаем обновленные переводы провайдеров
        providers = self._detect_available_providers()
        
        # Блокируем сигналы для предотвращения лишних вызовов
        self.ui.modelProviderCombo.blockSignals(True)
        self.ui.upscalerProviderCombo.blockSignals(True)
        
        # Обновляем провайдеров модели
        self.ui.modelProviderCombo.clear()
        for provider_id, provider_name in providers:
            self.ui.modelProviderCombo.addItem(provider_name, provider_id)
        
        # Обновляем провайдеров upscaler
        self.ui.upscalerProviderCombo.clear()
        for provider_id, provider_name in providers:
            self.ui.upscalerProviderCombo.addItem(provider_name, provider_id)
        
        # Восстанавливаем выбранные провайдеры
        for i in range(self.ui.modelProviderCombo.count()):
            if self.ui.modelProviderCombo.itemData(i) == current_model_provider:
                self.ui.modelProviderCombo.setCurrentIndex(i)
                break
                
        for i in range(self.ui.upscalerProviderCombo.count()):
            if self.ui.upscalerProviderCombo.itemData(i) == current_upscaler_provider:
                self.ui.upscalerProviderCombo.setCurrentIndex(i)
                break
        
        # Разблокируем сигналы
        self.ui.modelProviderCombo.blockSignals(False)
        self.ui.upscalerProviderCombo.blockSignals(False)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("LiveSwapping")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("LiveSwapping Team")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    gui = VideoGUI()
    gui.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 