#!/usr/bin/env python3
"""
Webcam Capture Dialog for ALIMA
Allows capturing and cropping images from webcam for LLM analysis
Claude Generated - Webcam Feature Implementation with StackedWidget
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QMessageBox,
    QRubberBand,
    QWidget,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QRectF, QPoint, QSize, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath
from PyQt6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget
import logging
from typing import Optional


class CropOverlayWidget(QWidget):
    """Overlay widget for crop rectangle selection - Claude Generated"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.crop_rect = None
        self.is_drawing = False  # Claude Generated: Drawing state tracking

        # Transparent background for overlay - Claude Generated
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        """Start crop rectangle selection - Claude Generated"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.crop_rect:
                # If already finalized, check if click is outside
                if not self.crop_rect.contains(event.pos()):
                    # Start new selection
                    self.crop_rect = None
                    self.is_drawing = True
                    self.start_point = event.pos()
                    self.end_point = event.pos()
                # If click inside, do nothing (keep existing selection)
            else:
                # Start new selection
                self.is_drawing = True
                self.start_point = event.pos()
                self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        """Update crop rectangle while dragging - Claude Generated"""
        if self.is_drawing and self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Finalize crop rectangle - Claude Generated"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.end_point = event.pos()

            # Create normalized rectangle
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            self.crop_rect = QRect(
                min(x1, x2), min(y1, y2),
                abs(x2 - x1), abs(y2 - y1)
            )
            self.update()

    def paintEvent(self, event):
        """Draw crop rectangle with transparent overlay - Claude Generated"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Determine active rectangle
        if self.crop_rect:
            rect = self.crop_rect
        elif self.is_drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            rect = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        else:
            return  # Nothing to draw

        # Draw semi-transparent overlay OUTSIDE crop area using QPainterPath (Donut shape)
        overlay_path = QPainterPath()
        overlay_path.addRect(QRectF(self.rect()))  # Full widget area (convert to QRectF)

        crop_path = QPainterPath()
        crop_path.addRect(QRectF(rect))  # Crop area (convert to QRectF)

        # Subtract crop area from overlay (creates donut shape)
        overlay_path = overlay_path.subtracted(crop_path)

        painter.fillPath(overlay_path, QColor(0, 0, 0, 120))

        # Draw crop rectangle border
        pen = QPen(QColor(0, 255, 0), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Draw corner handles
        handle_size = 10
        painter.setBrush(QColor(0, 255, 0))
        painter.setPen(Qt.PenStyle.NoPen)

        corners = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ]

        for x, y in corners:
            painter.drawEllipse(QPoint(x, y), handle_size, handle_size)

    def get_crop_rect(self) -> Optional[QRect]:
        """Get the finalized crop rectangle - Claude Generated"""
        return self.crop_rect

    def reset(self):
        """Reset crop selection - Claude Generated"""
        self.start_point = None
        self.end_point = None
        self.crop_rect = None
        self.is_drawing = False
        self.update()


class WebcamCaptureDialog(QDialog):
    """Dialog for capturing and cropping webcam images - Claude Generated"""

    # Signals
    image_captured = pyqtSignal(QImage)  # Emits the cropped image

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Camera components
        self.camera = None
        self.image_capture = None
        self.capture_session = None

        # State
        self.captured_image = None
        self.is_preview_mode = True  # True = live feed, False = captured image

        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Setup the dialog UI with StackedWidget - Claude Generated"""
        self.setWindowTitle("📷 Webcam Capture")
        self.setModal(True)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Kamera:"))

        self.camera_selector = QComboBox()
        camera_layout.addWidget(self.camera_selector)
        camera_layout.addStretch()

        layout.addLayout(camera_layout)

        # StackedWidget for switching between camera and image editing
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setMinimumSize(640, 480)
        
        # Page 0: Camera Preview
        self.camera_page = QWidget()
        camera_page_layout = QVBoxLayout(self.camera_page)
        camera_page_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_widget = QVideoWidget()
        camera_page_layout.addWidget(self.video_widget)
        
        self.stacked_widget.addWidget(self.camera_page)
        
        # Page 1: Image Editing with crop overlay
        self.editing_page = QWidget()
        editing_page_layout = QVBoxLayout(self.editing_page)
        editing_page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Container for image label and crop overlay
        self.image_container = QWidget()
        self.image_container.setMinimumSize(640, 480)
        
        # Image label for captured image
        self.image_label = QLabel(self.image_container)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        
        # Crop overlay for editing page
        self.crop_overlay = CropOverlayWidget(self.image_container)
        
        editing_page_layout.addWidget(self.image_container)
        self.stacked_widget.addWidget(self.editing_page)
        
        layout.addWidget(self.stacked_widget)

        # Status label
        self.status_label = QLabel("📹 Live-Vorschau aktiv")
        self.status_label.setStyleSheet("color: #2196f3; font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)

        # Instructions label
        self.instructions_label = QLabel("Klicken Sie auf 'Aufnahme', um ein Foto zu machen.")
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.capture_button = QPushButton("📷 Aufnahme")
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        self.capture_button.clicked.connect(self.capture_image)
        button_layout.addWidget(self.capture_button)

        self.retake_button = QPushButton("🔄 Neu aufnehmen")
        self.retake_button.setVisible(False)
        self.retake_button.clicked.connect(self.retake_image)
        button_layout.addWidget(self.retake_button)

        button_layout.addStretch()

        self.accept_button = QPushButton("✓ Übernehmen")
        self.accept_button.setEnabled(False)
        self.accept_button.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.accept_button.clicked.connect(self.accept_capture)
        button_layout.addWidget(self.accept_button)

        self.cancel_button = QPushButton("✗ Abbrechen")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        
        # Start with camera page
        self.stacked_widget.setCurrentIndex(0)

    def setup_camera(self):
        """Initialize camera and populate camera selector - Claude Generated"""
        try:
            # Get available cameras
            cameras = QMediaDevices.videoInputs()

            if not cameras:
                QMessageBox.warning(
                    self,
                    "Keine Kamera gefunden",
                    "Es wurde keine Kamera erkannt. Bitte schließen Sie eine Kamera an."
                )
                self.capture_button.setEnabled(False)
                return

            # Populate camera selector
            for camera in cameras:
                self.camera_selector.addItem(camera.description(), camera)

            # Connect camera selector
            self.camera_selector.currentIndexChanged.connect(self.change_camera)

            # Start with first camera
            self.start_camera(cameras[0])

        except Exception as e:
            self.logger.error(f"Error setting up camera: {e}")
            QMessageBox.critical(
                self,
                "Kamera-Fehler",
                f"Fehler beim Initialisieren der Kamera:\n{str(e)}"
            )

    def start_camera(self, camera_device):
        """Start camera with given device - Claude Generated"""
        try:
            # Stop existing camera if any
            if self.camera:
                self.camera.stop()

            # Create new camera
            self.camera = QCamera(camera_device)
            self.image_capture = QImageCapture(self.camera)
            self.capture_session = QMediaCaptureSession()

            # Configure image capture quality
            from PyQt6.QtMultimedia import QImageCapture as QIC
            self.image_capture.setQuality(QIC.Quality.VeryHighQuality)

            # Connect image capture signal
            self.image_capture.imageCaptured.connect(self.on_image_captured)
            self.image_capture.errorOccurred.connect(self.on_capture_error)

            # Setup capture session
            self.capture_session.setCamera(self.camera)
            self.capture_session.setImageCapture(self.image_capture)
            self.capture_session.setVideoOutput(self.video_widget)

            # Start camera
            self.camera.start()

            self.logger.info(f"Camera started: {camera_device.description()}")

        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            QMessageBox.critical(
                self,
                "Kamera-Fehler",
                f"Fehler beim Starten der Kamera:\n{str(e)}"
            )

    def change_camera(self, index):
        """Change to selected camera - Claude Generated"""
        if index < 0:
            return

        camera_device = self.camera_selector.itemData(index)
        if camera_device:
            self.start_camera(camera_device)

    def capture_image(self):
        """Capture image from camera - Claude Generated"""
        if not self.camera or not self.image_capture:
            QMessageBox.warning(self, "Fehler", "Kamera nicht bereit")
            return

        try:
            # Capture image
            self.image_capture.capture()
            self.logger.info("Image capture initiated")

        except Exception as e:
            self.logger.error(f"Error capturing image: {e}")
            QMessageBox.critical(self, "Fehler", f"Fehler bei der Bildaufnahme:\n{str(e)}")

    def on_image_captured(self, capture_id, image: QImage):
        """Handle captured image from camera - Claude Generated"""
        self.logger.info(f"Image captured (id={capture_id}): {image.width()}x{image.height()}")

        # Store captured image
        self.captured_image = image
        self.is_preview_mode = False
        
        # Stop camera to free resources
        #if self.camera:
        #    self.camera.stop()
        
        # Switch to editing page
        self.switch_to_editing_mode()

    def switch_to_editing_mode(self):
        """Switch to image editing mode - Claude Generated"""
        if not self.captured_image:
            return
            
        # Get container size for scaling
        container_size = self.image_container.size()
        if container_size.width() <= 0 or container_size.height() <= 0:
            container_size = QSize(640, 480)  # Fallback size
        
        # Display captured image in image label
        pixmap = QPixmap.fromImage(self.captured_image)
        scaled_pixmap = pixmap.scaled(
            container_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image in the container
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setGeometry(0, 0, container_size.width(), container_size.height())
        
        # Setup crop overlay to match image label
        self.crop_overlay.setGeometry(self.image_label.geometry())
        self.crop_overlay.reset()
        self.crop_overlay.show()
        self.crop_overlay.raise_()
        
        # Switch to editing page
        self.stacked_widget.setCurrentIndex(1)
        
        # Update UI
        self.status_label.setText("📸 Bild aufgenommen - Wählen Sie den Bereich aus")
        self.status_label.setStyleSheet("color: #ff9800; font-weight: bold; padding: 5px;")
        self.instructions_label.setText(
            "Ziehen Sie mit der Maus ein Rechteck um den gewünschten Bereich. "
            "Klicken Sie dann auf 'Übernehmen' oder 'Neu aufnehmen'."
        )
        
        # Update buttons
        self.capture_button.setVisible(False)
        self.retake_button.setVisible(True)
        self.accept_button.setEnabled(True)

    def switch_to_camera_mode(self):
        """Switch back to camera preview mode - Claude Generated"""
        # Switch to camera page
        self.stacked_widget.setCurrentIndex(0)
        
        # Hide crop overlay
        self.crop_overlay.hide()
        
        # Restart camera if needed
        if self.camera and not self.camera.isActive():
            self.camera.start()
        
        # Update UI
        self.status_label.setText("📹 Live-Vorschau aktiv")
        self.status_label.setStyleSheet("color: #2196f3; font-weight: bold; padding: 5px;")
        self.instructions_label.setText("Klicken Sie auf 'Aufnahme', um ein Foto zu machen.")
        
        # Update buttons
        self.capture_button.setVisible(True)
        self.retake_button.setVisible(False)
        self.accept_button.setEnabled(False)

    def on_capture_error(self, capture_id, error, error_string):
        """Handle capture error - Claude Generated"""
        self.logger.error(f"Capture error (id={capture_id}, code={error}): {error_string}")
        QMessageBox.critical(
            self,
            "Aufnahme-Fehler",
            f"Fehler bei der Bildaufnahme:\n{error_string}"
        )

    def retake_image(self):
        """Retake image (switch back to camera mode) - Claude Generated"""
        # Reset state
        self.captured_image = None
        self.is_preview_mode = True
        self.crop_overlay.reset()
        
        # Switch back to camera mode
        self.switch_to_camera_mode()

    def accept_capture(self):
        """Accept captured image and emit signal - Claude Generated"""
        if not self.captured_image:
            QMessageBox.warning(self, "Fehler", "Kein Bild vorhanden")
            return

        # Get crop rectangle if any
        crop_rect = self.crop_overlay.get_crop_rect()

        if crop_rect and crop_rect.width() > 10 and crop_rect.height() > 10:
            # Crop the image based on the rectangle
            # Need to scale crop_rect to match actual image dimensions
            display_size = self.image_label.size()
            pixmap = self.image_label.pixmap()
            
            if pixmap:
                pixmap_size = pixmap.size()
                
                # Calculate scaling factors
                scale_x = self.captured_image.width() / pixmap_size.width()
                scale_y = self.captured_image.height() / pixmap_size.height()
                
                # Calculate offset (image might be centered in label)
                offset_x = (display_size.width() - pixmap_size.width()) // 2
                offset_y = (display_size.height() - pixmap_size.height()) // 2
                
                # Adjust crop rectangle for offset and scaling
                adjusted_crop_rect = QRect(
                    int((crop_rect.x() - offset_x) * scale_x),
                    int((crop_rect.y() - offset_y) * scale_y),
                    int(crop_rect.width() * scale_x),
                    int(crop_rect.height() * scale_y)
                )
                
                # Ensure rectangle is within bounds
                adjusted_crop_rect = adjusted_crop_rect.intersected(
                    QRect(0, 0, self.captured_image.width(), self.captured_image.height())
                )
                
                final_image = self.captured_image.copy(adjusted_crop_rect)
                self.logger.info(f"Image cropped to: {final_image.width()}x{final_image.height()}")
            else:
                final_image = self.captured_image
                self.logger.info("Using full image (no pixmap)")
        else:
            # Use full image if no valid crop rectangle
            final_image = self.captured_image
            self.logger.info("Using full image (no crop)")

        # Show preview of final image before closing - Claude Generated
        self.show_final_preview(final_image)

        # Emit the captured/cropped image
        self.image_captured.emit(final_image)

        # Close dialog
        self.accept()

    def show_final_preview(self, final_image: QImage):
        """Show brief preview of final cropped image - Claude Generated"""
        from PyQt6.QtWidgets import QApplication
        import time

        # Hide crop overlay
        self.crop_overlay.hide()

        # Display final image
        preview_pixmap = QPixmap.fromImage(final_image)
        scaled_preview = preview_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_preview)

        # Update status
        self.status_label.setText("✓ Bild wird verarbeitet...")
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold; padding: 5px;")
        self.instructions_label.setText(f"Finale Bildgröße: {final_image.width()}x{final_image.height()} Pixel")

        # Force UI update and show for a brief moment
        QApplication.processEvents()
        time.sleep(0.5)  # Brief preview (500ms)

    def resizeEvent(self, event):
        """Handle resize events to update image and crop overlay - Claude Generated"""
        super().resizeEvent(event)

        # Update image container and crop overlay when editing page is active
        if (self.stacked_widget.currentIndex() == 1 and
            hasattr(self, 'crop_overlay') and
            hasattr(self, 'image_label') and
            hasattr(self, 'captured_image') and
            self.captured_image):

            # Update container and label geometry
            container_size = self.image_container.size()
            self.image_label.setGeometry(0, 0, container_size.width(), container_size.height())

            # Re-scale the captured image to new size
            pixmap = QPixmap.fromImage(self.captured_image)
            scaled_pixmap = pixmap.scaled(
                container_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

            # Update crop overlay to match
            self.crop_overlay.setGeometry(self.image_label.geometry())

            # Reset crop selection on resize (coordinates would be invalid)
            self.crop_overlay.reset()

    def closeEvent(self, event):
        """Cleanup on close - Claude Generated"""
        if self.camera:
            self.camera.stop()
        event.accept()