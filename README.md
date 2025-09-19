# Object Tracking Algorithm - Autobrains Assignment

A multi-object tracking algorithm that assigns consistent object IDs across consecutive frames in video sequences.

## Quick Start

### Prerequisites

- Python 3.7+
- Required packages: `pandas`, `numpy`, `opencv-python`, `matplotlib`

### Installation

1. **Install Python packages:**
```bash
pip install pandas numpy opencv-python matplotlib
```

2. **Or using conda:**
```bash
conda install pandas numpy opencv-python matplotlib
```

### Basic Usage

**Run the tracking algorithm:**
```bash
python Roni_Roitbord_tracker.py --input detections.tsv --output Roni_Roitbord_detections.tsv
```

**View results with interactive viewer:**
```bash
python interactive_viewer.py --tracker_file Roni_Roitbord_detections.tsv --images_dir images
```

## Command Line Options

### Tracking Script (`Roni_Roitbord_tracker.py`)

```bash
python Roni_Roitbord_tracker.py [OPTIONS]

Options:
  -i, --input FILE     Input detections TSV file (default: detections.tsv)
  -o, --output FILE    Output tracked detections TSV file (default: Roni_Roitbord_detections.tsv)
  -h, --help           Show help message
```

**Examples:**
```bash
# Use default filenames
python Roni_Roitbord_tracker.py

# Specify custom input/output files
python Roni_Roitbord_tracker.py --input my_detections.tsv --output my_results.tsv

# Short form
python Roni_Roitbord_tracker.py -i detections.tsv -o results.tsv
```

### Interactive Viewer (`interactive_viewer.py`)

```bash
python interactive_viewer.py [OPTIONS]

Options:
  --tracker_file FILE  Path to tracked detections TSV file
  --images_dir DIR     Directory containing input images
  -h, --help          Show help message
```

**Examples:**
```bash
# View tracking results
python interactive_viewer.py --tracker_file Roni_Roitbord_detections.tsv --images_dir images

# View with custom files
python interactive_viewer.py --tracker_file my_results.tsv --images_dir my_images
```

## File Structure

```
project/
├── Roni_Roitbord_tracker.py      # Main tracking algorithm
├── interactive_viewer.py          # Visualization tool
├── detections.tsv                 # Input detections file
├── images/                        # Input images directory
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── Roni_Roitbord_detections.tsv  # Output tracking results
├── README.md                      # This file
└── SOLUTION_DOCUMENTATION.md     # Technical documentation
```

## Input Format

The input TSV file should have the following columns:
- `name`: Image filename
- `x_center`: X coordinate of bounding box center
- `y_center`: Y coordinate of bounding box center  
- `width`: Width of bounding box
- `height`: Height of bounding box
- `label`: Object class (e.g., CAR, PEDESTRIAN, TRIANGLE_ROAD_SIGN)

## Output Format

The output TSV file includes all input columns plus:
- `object_id`: Unique identifier for each tracked object

## Interactive Viewer Controls

| Key | Action |
|-----|--------|
| `←` `→` | Previous/Next frame |
| `A` `D` | Alternative navigation |
| `Home` | First frame |
| `End` | Last frame |
| `Space` | Next frame |
| `Enter` | Next frame |
| `Q` | Quit viewer |
| `S` | Save current view |

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'cv2'**
```bash
pip install opencv-python
# or
conda install opencv
```

**2. ModuleNotFoundError: No module named 'pandas'**
```bash
pip install pandas numpy
# or
conda install pandas numpy
```

**3. Python interpreter issues**
If using conda environment:
```bash
/opt/anaconda3/envs/cv/bin/python interactive_viewer.py --tracker_file Roni_Roitbord_detections.tsv --images_dir images
```

**4. File not found errors**
- Ensure input files exist in the correct directory
- Check file paths are correct
- Verify TSV file format matches expected structure

### Performance Notes

- **Processing time**: ~2-3 seconds for 484 frames
- **Memory usage**: ~50-100MB for typical datasets
- **Output size**: Typically 15-20% smaller than input (duplicates removed)

## Algorithm Parameters

The tracking algorithm uses these key parameters (configurable in code):

- `iou_threshold`: 0.3 (minimum IoU for spatial overlap)
- `distance_threshold`: 200px (maximum distance for matching)
- `max_track_age`: 20 frames (inactive track retention)

## Example Workflow

1. **Prepare your data:**
   - Ensure `detections.tsv` is in the project directory
   - Place images in `images/` directory

2. **Run tracking:**
   ```bash
   python Roni_Roitbord_tracker.py
   ```

3. **View results:**
   ```bash
   python interactive_viewer.py --tracker_file Roni_Roitbord_detections.tsv --images_dir images
   ```

4. **Navigate through frames:**
   - Use arrow keys to move between frames
   - Observe object IDs remain consistent across frames
   - Press `Q` to quit

## Support

For technical details and algorithm explanation, see `SOLUTION_DOCUMENTATION.md`.

For issues or questions, refer to the troubleshooting section above.