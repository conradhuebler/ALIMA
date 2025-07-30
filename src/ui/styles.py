"""
Unified UI styles for ALIMA application
"""

# Color palette
COLORS = {
    "primary": "#4a86e8",  # Blue
    "primary_hover": "#3a76d8",
    "primary_pressed": "#2a66c8",
    "secondary": "#6aa84f",  # Green
    "secondary_hover": "#5a984f",
    "secondary_pressed": "#4a883f",
    "accent": "#f1c232",  # Gold
    "accent_hover": "#e1b222",
    "accent_pressed": "#d1a212",
    "background": "#ffffff",
    "background_light": "#f8f9fa",
    "background_dark": "#e9ecef",
    "text": "#333333",
    "text_light": "#666666",
    "text_muted": "#999999",
    "error": "#e74c3c",
    "error_hover": "#d73527",
    "success": "#28a745",
    "success_hover": "#218838",
    "warning": "#ffc107",
    "warning_hover": "#e0a800",
    "border": "#cccccc",
    "border_light": "#e0e0e0",
    "border_dark": "#999999",
}


def get_main_stylesheet():
    """Get the main application stylesheet"""
    return f"""
    /* Main application styles */
    QWidget {{
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
        color: {COLORS['text']};
    }}
    
    /* Group boxes */
    QGroupBox {{
        font-weight: bold;
        font-size: 11pt;
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 10px;
        background-color: {COLORS['background_light']};
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        background-color: {COLORS['background_light']};
        color: {COLORS['text']};
    }}
    
    /* Primary buttons */
    QPushButton {{
        border: none;
        border-radius: 2px;
        padding: 2px 5px;
        background-color: {COLORS['primary']};
        color: white;
        font-weight: bold;
        font-size: 10pt;
        min-height: 20px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['primary_hover']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['primary_pressed']};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS['background_dark']};
        color: {COLORS['text_muted']};
    }}
    
    /* Text inputs */
    QTextEdit {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        background-color: {COLORS['background']};
        font-size: 11pt;
        line-height: 1.4;
    }}
    
    QTextEdit:focus {{
        border: 2px solid {COLORS['primary']};
    }}
    
    QLineEdit {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 8px;
        background-color: {COLORS['background']};
        font-size: 10pt;
    }}
    
    QLineEdit:focus {{
        border: 2px solid {COLORS['primary']};
    }}
    
    /* Labels */
    QLabel {{
        color: {COLORS['text']};
        font-size: 10pt;
    }}
    
    /* Combo boxes */
    QComboBox {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 6px 8px;
        background-color: {COLORS['background']};
        font-size: 10pt;
        min-height: 20px;
    }}
    
    QComboBox:focus {{
        border: 2px solid {COLORS['primary']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox::down-arrow {{
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iNiIgdmlld0JveD0iMCAwIDEwIDYiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik01IDZMMCAwaDEwTDUgNloiIGZpbGw9IiM2NjY2NjYiLz4KPC9zdmc+);
        width: 10px;
        height: 6px;
    }}
    
    /* Checkboxes */
    QCheckBox {{
        color: {COLORS['text']};
        spacing: 8px;
        font-size: 10pt;
    }}
    
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {COLORS['border']};
        border-radius: 3px;
        background-color: {COLORS['background']};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
    }}
    
    /* Tables */
    QTableWidget {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        background-color: {COLORS['background']};
        gridline-color: {COLORS['border_light']};
        font-size: 10pt;
    }}
    
    QTableWidget::item {{
        padding: 8px;
        border-bottom: 1px solid {COLORS['border_light']};
    }}
    
    QTableWidget::item:selected {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    QHeaderView::section {{
        background-color: {COLORS['background_light']};
        padding: 8px;
        border: 1px solid {COLORS['border']};
        font-weight: bold;
    }}
    
    /* Progress bars */
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        text-align: center;
        font-size: 10pt;
        font-weight: bold;
    }}
    
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 3px;
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {COLORS['background_dark']};
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
        width: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {COLORS['primary_hover']};
    }}
    
    /* Spin boxes */
    QSpinBox {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 6px 8px;
        background-color: {COLORS['background']};
        font-size: 10pt;
    }}
    
    QSpinBox:focus {{
        border: 2px solid {COLORS['primary']};
    }}
    
    /* Tab widgets */
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        background-color: {COLORS['background']};
    }}
    
    QTabBar::tab {{
        background-color: {COLORS['background_light']};
        border: 1px solid {COLORS['border']};
        padding: 8px 16px;
        margin-right: 2px;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {COLORS['background']};
        border-bottom: 2px solid {COLORS['primary']};
    }}
    
    QTabBar::tab:hover {{
        background-color: {COLORS['background_dark']};
    }}
    
    /* Splitters */
    QSplitter::handle {{
        background-color: {COLORS['background_dark']};
        border: 1px solid {COLORS['border']};
    }}
    
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    
    /* Tree widgets */
    QTreeWidget {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        background-color: {COLORS['background']};
        font-size: 10pt;
    }}
    
    QTreeWidget::item {{
        padding: 4px;
        border-bottom: 1px solid {COLORS['border_light']};
    }}
    
    QTreeWidget::item:selected {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    QTreeWidget::item:hover {{
        background-color: {COLORS['background_light']};
    }}
    """


def get_button_styles():
    """Get button style variations"""
    return {
        "primary": f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
        "secondary": f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['secondary_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
        "accent": f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['text']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
        "success": f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
        "warning": f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['warning_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['warning_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
        "error": f"""
            QPushButton {{
                background-color: {COLORS['error']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['error_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['error_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['background_dark']};
                color: {COLORS['text_muted']};
            }}
        """,
    }


def get_status_label_styles():
    """Get status label styles"""
    return {
        "default": f"color: {COLORS['text']};",
        "success": f"color: {COLORS['success']}; font-weight: bold;",
        "warning": f"color: {COLORS['warning']}; font-weight: bold;",
        "error": f"color: {COLORS['error']}; font-weight: bold;",
        "info": f"color: {COLORS['primary']}; font-weight: bold;",
        "muted": f"color: {COLORS['text_muted']};",
    }


def get_image_preview_style():
    """Get image preview container style"""
    return f"""
        QLabel {{
            border: 2px dashed {COLORS['border']};
            border-radius: 8px;
            background-color: {COLORS['background_light']};
            color: {COLORS['text_muted']};
            font-size: 12pt;
            text-align: center;
            padding: 20px;
        }}
    """
