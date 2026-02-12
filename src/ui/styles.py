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
    "shadow": "rgba(0, 0, 0, 0.1)",
    # Confidence level colors (text, background) - Claude Generated
    "confidence_very_high_text": "#2d5016",
    "confidence_very_high_bg": "#d4edda",
    "confidence_high_text": "#0c5460",
    "confidence_high_bg": "#d1ecf1",
    "confidence_medium_text": "#664d03",
    "confidence_medium_bg": "#fff3cd",
    "confidence_low_text": "#721c24",
    "confidence_low_bg": "#f8d7da",
}

# Standard layout constants
LAYOUT = {
    "margin": 15,
    "spacing": 12,
    "inner_spacing": 8,
    "border_radius": 8,
    "input_font_size": 11,
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

    /* Scrollbars */
    QScrollBar:vertical {{
        border: none;
        background: {COLORS['background_light']};
        width: 10px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background: {COLORS['border']};
        min-height: 20px;
        border-radius: 5px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {COLORS['border_dark']};
    }}

    QScrollBar:horizontal {{
        border: none;
        background: {COLORS['background_light']};
        height: 10px;
        margin: 0px;
    }}

    QScrollBar::handle:horizontal {{
        background: {COLORS['border']};
        min-width: 20px;
        border-radius: 5px;
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


def get_confidence_style(count: int) -> tuple:
    """Get color styling based on confidence/frequency count - Claude Generated

    Args:
        count: Number of catalog hits or unique titles

    Returns:
        Tuple of (text_color, bg_color, label, bar)
        where bar is emoji confidence indicator
    """
    if count > 50:
        return (
            COLORS["confidence_very_high_text"],
            COLORS["confidence_very_high_bg"],
            "Very High",
            "\U0001f7e9" * 5,
        )
    elif count > 20:
        return (
            COLORS["confidence_high_text"],
            COLORS["confidence_high_bg"],
            "High",
            "\U0001f7e9" * 3,
        )
    elif count > 5:
        return (
            COLORS["confidence_medium_text"],
            COLORS["confidence_medium_bg"],
            "Medium",
            "\U0001f7e9" * 2,
        )
    else:
        return (
            COLORS["confidence_low_text"],
            COLORS["confidence_low_bg"],
            "Low",
            "\U0001f7e9",
        )


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
