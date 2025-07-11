# -*- coding: utf-8 -*-
"""Красивый Qt-GUI для реал-тайм обработки на основе скомпилированного .ui файла."""

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
from .realtime_gui_ui import Ui_RealtimeGUI

# Локализация
import liveswapping.utils.localisation as loc

from liveswapping.run import run as cli_run
from liveswapping.ai_models.download_models import MODELS, ensure_model


class RealtimeWorker(threading.Thread):
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
            
            # Запускаем процесс как подпроцесс для возможности его остановки
            # Используем корневой run.py вместо -m liveswapping.run для избежания проблем с Python path
            run_script = Path(__file__).parent.parent.parent / "run.py"
            cmd = [sys.executable, str(run_script)] + self.args
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            
            # Ждем завершения процесса или сигнала остановки
            while self._process.poll() is None:
                if self._stop_event.wait(timeout=0.1):
                    # Получили сигнал остановки
                    self._terminate_process()
                    break
            
            # Проверяем код возврата
            if self._process and self._process.returncode not in [0, None]:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                if stderr_output and not self._stop_event.is_set():
                    raise Exception(f"Process failed with code {self._process.returncode}: {stderr_output}")
                    
        except Exception as e:
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


class RealtimeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: RealtimeWorker | None = None
        
        # Загрузка UI
        self.ui = Ui_RealtimeGUI()
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
        
        # Увеличиваем минимальную ширину чтобы текст не обрезался
        self.ui.languageCombo.setMinimumWidth(170)
        
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
        
        QSpinBox, QDoubleSpinBox {
            border: 2px solid #1a4480;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #0f1419;
            color: #e0f7ff;
            min-width: 80px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #00bcd4;
        }
        
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background-color: #1565c0;
            border: none;
            width: 20px;
        }
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #1976d2;
        }
        
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            width: 0;
            height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 6px solid #e0f7ff;
        }
        
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
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
        
        # Модели уже настроены в .ui файле
        
        # Устанавливаем значения по умолчанию
        self.ui.resolutionSpinBox.setValue(128)
        self.ui.delaySpinBox.setValue(0)

    def _connect_signals(self):
        """Подключение сигналов."""
        self.ui.sourceBrowseButton.clicked.connect(self._browse_source)
        self.ui.modelBrowseButton.clicked.connect(self._browse_model)
        self.ui.attributeBrowseButton.clicked.connect(self._browse_attribute)
        self.ui.modelCombo.currentIndexChanged.connect(self._model_selected)
        self.ui.startButton.clicked.connect(self._start)
        self.ui.stopButton.clicked.connect(self._stop)
        self.ui.languageCombo.currentIndexChanged.connect(self._change_language)

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

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            loc.get("dialog_select_model"), 
            "models", 
            loc.get("filter_models")
        )
        if path:
            self.ui.modelPathEdit.setText(path)

    def _browse_attribute(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select attribute direction file", 
            "", 
            "Numpy files (*.npy)"
        )
        if path:
            self.ui.attributeEdit.setText(path)

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
    # Processing control
    # ------------------------------------------------------------------
    def _start(self):
        if self._worker:
            QMessageBox.warning(self, loc.get("already_running"), loc.get("processing_in_progress"))
            return
            
        # Validate inputs
        src_path = Path(self.ui.sourceEdit.text())
        model_path = Path(self.ui.modelPathEdit.text())
        
        if not src_path.exists():
            QMessageBox.warning(self, loc.get("missing_source"), loc.get("select_valid_source"))
            return
        if not model_path.exists():
            QMessageBox.warning(self, loc.get("missing_model"), loc.get("select_valid_model"))
            return

        # Compose arguments
        args = [
            "realtime",
            "--source", str(src_path),
            "--modelPath", str(model_path),
            "--resolution", str(self.ui.resolutionSpinBox.value()),
            "--delay", str(self.ui.delaySpinBox.value()),
        ]
        
        # Add attribute direction if specified
        if self.ui.attributeEdit.text():
            args.extend(["--attribute_dir", self.ui.attributeEdit.text()])
            args.extend(["--attribute_steps", str(self.ui.stepsSpinBox.value())])
        
        # Add options
        if self.ui.obsCheckBox.isChecked():
            args.append("--obs")
        if self.ui.fpsCheckBox.isChecked():
            args.append("--show_fps")
        if self.ui.enhanceCheckBox.isChecked():
            args.append("--enhance")

        # Start processing
        self._worker = RealtimeWorker(args)
        self._worker.start()
        
        # Update UI
        self.ui.startButton.setEnabled(False)
        self.ui.stopButton.setEnabled(True)
        self.ui.infoLabel.setText(loc.get("info_stop"))
        
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
            return
            
        self._worker.join()
        
        # Update UI on main thread
        QTimer.singleShot(0, self._on_worker_finished)

    def _on_worker_finished(self):
        """Called when worker finishes - updates UI."""
        if self._worker and self._worker.exc:
            QMessageBox.critical(
                self, 
                loc.get("processing_error"), 
                f"{loc.get('error_occurred')}\n\n{str(self._worker.exc)}"
            )
        elif self._worker and self._worker._stop_event.is_set():
            # Процесс был остановлен пользователем - не показываем сообщение
            self._reset_ui()
            return
        else:
            # Процесс завершился успешно
            QMessageBox.information(
                self, 
                loc.get("success"), 
                "Realtime processing completed successfully!"
            )
        self._reset_ui()

    def _reset_ui(self):
        """Resets UI to initial state."""
        self._worker = None
        self.ui.startButton.setEnabled(True)
        self.ui.stopButton.setEnabled(False)
        self.ui.infoLabel.setText(loc.get("info_stop"))

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
        self.setWindowTitle(loc.get("title_realtime"))
        
        # Обновляем группы и лейблы
        self.ui.sourceGroup.setTitle(loc.get("source_image"))
        self.ui.modelGroup.setTitle(loc.get("ai_model"))
        self.ui.processingGroup.setTitle(loc.get("main_params"))
        self.ui.attributesGroup.setTitle(loc.get("advanced_params"))
        self.ui.optionsGroup.setTitle(loc.get("options"))
        
        # Обновляем лейблы полей
        self.ui.sourceLabel.setText(loc.get("source_image"))
        self.ui.modelLabel.setText(loc.get("ai_model"))
        self.ui.modelPathLabel.setText(loc.get("model_path"))
        self.ui.resolutionLabel.setText(loc.get("resolution"))
        self.ui.delayLabel.setText(loc.get("delay"))
        self.ui.attributeLabel.setText(loc.get("attribute_dir"))
        self.ui.stepsLabel.setText(loc.get("attribute_steps"))
        
        # Обновляем кнопки
        self.ui.sourceBrowseButton.setText(loc.get("browse"))
        self.ui.modelBrowseButton.setText(loc.get("browse"))
        self.ui.attributeBrowseButton.setText(loc.get("browse"))
        self.ui.startButton.setText(loc.get("start"))
        self.ui.stopButton.setText(loc.get("stop"))
        
        # Обновляем чекбоксы
        self.ui.mouthCheckBox.setText(loc.get("retain_target_mouth"))
        self.ui.obsCheckBox.setText(loc.get("send_to_obs"))
        self.ui.fpsCheckBox.setText(loc.get("show_fps_delay"))
        self.ui.enhanceCheckBox.setText(loc.get("enhance_res_cam"))
        
        # Обновляем информационный лейбл
        self.ui.infoLabel.setText(loc.get("info_stop"))


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
    
    gui = RealtimeGUI()
    gui.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 