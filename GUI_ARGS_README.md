# GUI-Enabled Argument Parser

This project now includes a custom `GuiArgumentParser` that provides both command-line and graphical user interfaces for your applications.

## Features

- **Dual Interface**: Applications can be run from command line with arguments OR launched as desktop apps with GUI dialogs
- **Automatic Detection**: Automatically detects whether CLI arguments are provided and switches modes accordingly
- **Smart Widgets**: Creates appropriate input widgets based on argument types:
  - File picker buttons for `argparse.FileType` arguments
  - Number inputs for int/float types
  - Checkboxes for boolean flags (`store_true`/`store_false`)
  - Dropdown menus for choice arguments
  - Text inputs for string arguments
- **Organized Layout**: Separates required and optional arguments into distinct groups
- **Validation**: Validates required arguments and type conversions

## Usage

### Converting Existing Applications

Replace `argparse.ArgumentParser()` with `GuiArgumentParser()`:

```python
# Before
import argparse
parser = argparse.ArgumentParser()

# After  
from microseg.utils.args import GuiArgumentParser
parser = GuiArgumentParser()
```

That's it! No other changes needed.

### Example

```python
from microseg.utils.args import GuiArgumentParser
import argparse

parser = GuiArgumentParser(description="My Application")
parser.add_argument('file', type=argparse.FileType('r'), help='Input image file')
parser.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file')
parser.add_argument('-d', '--desc', default='output', help='Description')
parser.add_argument('-c', '--channel', type=int, default=0, help='Channel number')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()
```

### Running Applications

**Command Line Mode** (traditional):
```bash
python -m microseg.image.segment /path/to/image.tif -d polygons
```

**GUI Mode** (no arguments):
```bash
python -m microseg.image.segment
```

This will open a dialog where users can:
- Click file picker buttons to browse for files (for `argparse.FileType` arguments)
- Enter text values in line edits
- Adjust numeric values with spinboxes
- Toggle boolean options with checkboxes
- Select from dropdown menus for choice arguments

The dialog organizes arguments into "Required Arguments" and "Optional Arguments" groups for better usability.

## Creating Desktop Applications

### 1. Create a Launcher Script

Use the provided `launch_image_segment.py` as a template:

```python
#!/usr/bin/env python3
import sys
import subprocess

cmd = [sys.executable, '-m', 'microseg.image.segment']
subprocess.run(cmd)
```

### 2. Make it Executable

```bash
chmod +x launch_image_segment.py
```

### 3. Create Desktop Integration

**Linux (.desktop file)**:
```ini
[Desktop Entry]
Name=Image Segmentor
Comment=Segment images with polygons
Exec=/path/to/launch_image_segment.py
Icon=/path/to/icon.png
Type=Application
Categories=Graphics;Science;
```

**macOS (.app bundle)**:
Create an application bundle or use tools like `py2app` to package the launcher.

**Windows (.bat file)**:
```batch
@echo off
python launch_image_segment.py
pause
```

## File Handling

File arguments should use `argparse.FileType` for proper handling:

```python
# For input files
parser.add_argument('input', type=argparse.FileType('r'), help='Input file')

# For output files  
parser.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file')

# For directories (detected from help text)
parser.add_argument('folder', help='Input folder or directory')
```

The GUI will show file picker buttons for `FileType` arguments and directory pickers when "folder" or "directory" appears in the help text.

## Testing

Test the functionality using the built-in test:

```bash
# Test CLI mode
python -m microseg.utils.args /path/to/file.txt -d test

# Test GUI mode  
python -m microseg.utils.args
```

## Implementation Details

The `GuiArgumentParser` works by:

1. Checking if command-line arguments are provided
2. If yes: behaves exactly like standard `ArgumentParser`
3. If no: creates a PyQt dialog with appropriate widgets for each argument
4. Validates input and returns a standard `argparse.Namespace` object

This ensures complete compatibility with existing code while adding GUI functionality.

## Updated Applications

The following applications have been updated to use `GuiArgumentParser`:

- `src/microseg/image/segment.py` - Image segmentation tool

To update other applications, simply replace the import and class name as shown above. 