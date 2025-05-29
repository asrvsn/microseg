'''
GUI-enabled ArgumentParser for microseg applications

This module provides a custom ArgumentParser that can operate in two modes:
1. CLI mode: Normal argparse behavior when command-line arguments are provided
2. GUI mode: Opens a PyQt dialog for argument input when no CLI arguments are provided
'''

import sys
import argparse
import os
from typing import Any, Dict, List, Optional, Union
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QLineEdit, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QComboBox, QTextEdit, QDialogButtonBox
)
from qtpy.QtCore import Qt


class ArgumentDialog(QDialog):
    """
    Dialog for collecting command-line arguments through a GUI interface.
    """
    
    def __init__(self, parser: argparse.ArgumentParser, parent=None):
        super().__init__(parent)
        self.parser = parser
        self.widgets = {}
        self.values = {}
        
        # Use parser prog name with "Arguments" appended, or default
        if parser.prog and parser.prog != 'argparse.py':
            # Use the program name if it's meaningful
            window_title = f'{parser.prog} Arguments'
        else:
            window_title = "Application Arguments"
            
        self.setWindowTitle(window_title)
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the dialog UI based on the parser's arguments."""
        layout = QVBoxLayout(self)
        
        # Add description at the top if it exists
        if self.parser.description:
            desc_label = QLabel(self.parser.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px;")
            layout.addWidget(desc_label)
        
        # Create scroll area for arguments
        scroll = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Separate required and optional arguments
        required_actions = []
        optional_actions = []
        
        for action in self.parser._actions:
            if action.dest in ['help', 'version']:
                continue
            if action.required or not action.option_strings:  # Positional args are required
                required_actions.append(action)
            else:
                optional_actions.append(action)
        
        # Create required arguments group
        if required_actions:
            required_group = QtWidgets.QGroupBox("Required Arguments")
            required_form = QFormLayout()
            
            for action in required_actions:
                widget = self._create_widget_for_action(action)
                if widget:
                    label = self._create_label_for_action(action)
                    required_form.addRow(label, widget)
                    self.widgets[action.dest] = widget
            
            required_group.setLayout(required_form)
            scroll_layout.addWidget(required_group)
        
        # Create optional arguments group
        if optional_actions:
            optional_group = QtWidgets.QGroupBox("Optional Arguments")
            optional_form = QFormLayout()
            
            for action in optional_actions:
                widget = self._create_widget_for_action(action)
                if widget:
                    label = self._create_label_for_action(action)
                    optional_form.addRow(label, widget)
                    self.widgets[action.dest] = widget
            
            optional_group.setLayout(optional_form)
            scroll_layout.addWidget(optional_group)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _create_label_for_action(self, action) -> QLabel:
        """Create a label for an argument action."""
        text = action.dest.replace('_', ' ').title()
        
        label = QLabel(text)
        if action.help:
            label.setToolTip(action.help)
        
        return label
        
    def _create_widget_for_action(self, action) -> Optional[QtWidgets.QWidget]:
        """Create an appropriate widget for an argument action."""
        if isinstance(action.type, argparse.FileType):
            return self._create_file_widget(action)
        elif action.type == int:
            return self._create_int_widget(action)
        elif action.type == float:
            return self._create_float_widget(action)
        elif isinstance(action, argparse._StoreTrueAction):
            return self._create_bool_widget(action)
        elif isinstance(action, argparse._StoreFalseAction):
            return self._create_bool_widget(action, default_checked=True)
        elif action.choices:
            return self._create_choice_widget(action)
        else:
            return self._create_string_widget(action)
            
    def _create_file_widget(self, action) -> QPushButton:
        """Create a file picker widget."""
        button = QPushButton()
        
        # Set initial button text
        if action.default and action.default != argparse.SUPPRESS:
            button.setText(str(action.default))
            button._file_path = str(action.default)
        else:
            button.setText("Browse...")
            button._file_path = ""
            
        button.clicked.connect(
            lambda: self._browse_file(button, action)
        )
        
        return button
        
    def _create_int_widget(self, action) -> QSpinBox:
        """Create an integer input widget."""
        widget = QSpinBox()
        widget.setRange(-2147483648, 2147483647)  # int32 range
        if action.default is not None and action.default != argparse.SUPPRESS:
            widget.setValue(action.default)
        return widget
        
    def _create_float_widget(self, action) -> QDoubleSpinBox:
        """Create a float input widget."""
        widget = QDoubleSpinBox()
        widget.setRange(-1e10, 1e10)
        widget.setDecimals(6)
        if action.default is not None and action.default != argparse.SUPPRESS:
            widget.setValue(action.default)
        return widget
        
    def _create_bool_widget(self, action, default_checked=False) -> QCheckBox:
        """Create a boolean input widget."""
        widget = QCheckBox()
        if action.default is not None and action.default != argparse.SUPPRESS:
            widget.setChecked(action.default)
        else:
            widget.setChecked(default_checked)
        return widget
        
    def _create_choice_widget(self, action) -> QComboBox:
        """Create a choice selection widget."""
        widget = QComboBox()
        widget.addItems([str(choice) for choice in action.choices])
        if action.default is not None and action.default != argparse.SUPPRESS:
            index = widget.findText(str(action.default))
            if index >= 0:
                widget.setCurrentIndex(index)
        return widget
        
    def _create_string_widget(self, action) -> QLineEdit:
        """Create a string input widget."""
        widget = QLineEdit()
        if action.default is not None and action.default != argparse.SUPPRESS:
            widget.setText(str(action.default))
        return widget
        
    def _browse_file(self, button: QPushButton, action):
        """Open file browser dialog."""
        current_path = getattr(button, '_file_path', '') or os.getcwd()
        
        # Determine if we're looking for a file or directory
        help_text = action.help.lower() if action.help else ""
        
        if "folder" in help_text or "directory" in help_text:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {action.dest}", current_path
            )
        else:
            # Use generic file dialog
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {action.dest}", current_path, "All Files (*)"
            )
            
        if path:
            button.setText(os.path.basename(path))
            button._file_path = path
            
    def get_values(self) -> Dict[str, Any]:
        """Extract values from all widgets."""
        values = {}
        
        for dest, widget in self.widgets.items():
            action = next(a for a in self.parser._actions if a.dest == dest)
            
            # Handle different widget types
            if isinstance(widget, QPushButton) and hasattr(widget, '_file_path'):
                # File picker button - return the path string for FileType processing
                value = getattr(widget, '_file_path', '')
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
                # Convert back to original type if needed
                if action.type and action.type != str:
                    try:
                        value = action.type(value)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(widget, QLineEdit):
                value = widget.text()
                # Convert to appropriate type
                if action.type and value:
                    try:
                        value = action.type(value)
                    except (ValueError, TypeError):
                        pass
            else:
                continue
                
            # Only include non-empty values or required arguments
            if value or action.required or isinstance(value, bool):
                values[dest] = value
                
        return values


class GuiArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that can show a GUI dialog when no CLI arguments are provided.
    
    Usage:
        parser = GuiArgumentParser(prog="My Application")
        parser.add_argument('file', help='Input file')
        parser.add_argument('-d', '--desc', default='output', help='Description')
        args = parser.parse_args()
    
    If command-line arguments are provided, behaves like normal ArgumentParser.
    If no arguments are provided, opens a GUI dialog for input.
    """
    
    def __init__(self, *args, title: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gui_mode = False
        self._gui_title = title
        
    def parse_args(self, args=None, namespace=None):
        """
        Parse arguments, using GUI if no CLI arguments are provided.
        """
        # If args is explicitly provided, use normal parsing
        if args is not None:
            return super().parse_args(args, namespace)
            
        # Check if we have any command-line arguments (excluding script name)
        cli_args = sys.argv[1:]
        
        # If we have CLI arguments, use normal parsing
        if cli_args:
            return super().parse_args(cli_args, namespace)
            
        # No CLI arguments - use GUI mode
        return self._parse_args_gui(namespace)
        
    def _parse_args_gui(self, namespace=None):
        """Parse arguments using GUI dialog."""
        # Ensure we have a QApplication
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            
        # Create and show dialog
        dialog = ArgumentDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            
            # Create namespace with values
            if namespace is None:
                namespace = argparse.Namespace()
                
            # Set defaults first
            for action in self._actions:
                if action.dest != 'help' and hasattr(action, 'default'):
                    if action.default != argparse.SUPPRESS:
                        setattr(namespace, action.dest, action.default)
                        
            # Override with user values, handling FileType conversion
            for dest, value in values.items():
                action = next(a for a in self._actions if a.dest == dest)
                
                # Handle FileType conversion
                if isinstance(action.type, argparse.FileType) and isinstance(value, str) and value:
                    try:
                        # Convert file path to file object using FileType
                        file_obj = action.type(value)
                        setattr(namespace, dest, file_obj)
                    except (IOError, OSError) as e:
                        self.error(f"can't open '{value}': {e}")
                else:
                    setattr(namespace, dest, value)
                
            # Validate required arguments
            for action in self._actions:
                if action.required and not hasattr(namespace, action.dest):
                    self.error(f"argument {action.dest} is required")
                    
            self._gui_mode = True
            return namespace
        else:
            # User cancelled - exit gracefully
            sys.exit(0)
            
    def is_gui_mode(self) -> bool:
        """Return True if arguments were parsed via GUI."""
        return self._gui_mode


if __name__ == '__main__':
    """
    Test script for GuiArgumentParser functionality.
    
    Run with arguments: python -m microseg.utils.args /path/to/file.txt -d test
    Run without arguments: python -m microseg.utils.args
    """
    
    def main():
        parser = GuiArgumentParser(prog="Test Application", description="Test GUI-enabled argument parser")
        
        # Add various types of arguments to test the GUI
        parser.add_argument('file', type=argparse.FileType('r'), help='Path to input file')
        parser.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file')
        parser.add_argument('-d', '--desc', type=str, default='test', help='Description string')
        parser.add_argument('-c', '--channel', type=int, default=0, help='Channel number')
        parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scale factor')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('--mode', choices=['fast', 'accurate', 'balanced'], default='balanced', help='Processing mode')
        parser.add_argument('--ignore-existing', action='store_true', help='Ignore existing files')
        
        args = parser.parse_args()
        
        print("Parsed arguments:")
        print(f"  File: {args.file.name if hasattr(args.file, 'name') else args.file}")
        print(f"  Output: {args.output.name if hasattr(args.output, 'name') else args.output}")
        print(f"  Description: {args.desc}")
        print(f"  Channel: {args.channel}")
        print(f"  Scale: {args.scale}")
        print(f"  Verbose: {args.verbose}")
        print(f"  Mode: {args.mode}")
        print(f"  Ignore existing: {args.ignore_existing}")
        print(f"  GUI mode: {parser.is_gui_mode()}")
        
        # Close file handles if they were opened
        if hasattr(args.file, 'close'):
            args.file.close()
        if hasattr(args.output, 'close'):
            args.output.close()

    main() 