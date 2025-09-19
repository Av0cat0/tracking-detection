#!/usr/bin/env python3
"""
Interactive viewer for tracking results with keyboard shortcuts.
Navigate through frames and see object IDs changing/persisting.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import cv2
import numpy as np
from typing import Dict, List, Optional
import argparse

class InteractiveTrackingViewer:
    def __init__(self, tracker_file: str, images_dir: str):
        """
        Initialize the interactive viewer.
        
        Args:
            tracker_file: Path to tracking results TSV file
            images_dir: Directory containing images
        """
        self.tracker_file = tracker_file
        self.images_dir = images_dir
        
        # Load tracking results
        self.results_df = self.load_results()
        if self.results_df.empty:
            print("No tracking results found!")
            return
        
        # Get all unique frames
        self.all_frames = sorted(self.results_df['name'].unique())
        self.current_frame_idx = 0
        self.current_frame = self.all_frames[0]
        
        # Colors for different object IDs
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Object ID history for visualization
        self.object_history = self.build_object_history()
        
        # Setup the plot
        self.setup_plot()
        
        # Print instructions
        self.print_instructions()
        
    def load_results(self) -> pd.DataFrame:
        """Load tracking results from TSV file."""
        if not os.path.exists(self.tracker_file):
            print(f"Tracker file not found: {self.tracker_file}")
            return pd.DataFrame()
        
        try:
            results_df = pd.read_csv(self.tracker_file, sep='\t')
            print(f"Loaded {len(results_df)} tracking results from {self.tracker_file}")
            return results_df
        except Exception as e:
            print(f"Error loading tracker file: {e}")
            return pd.DataFrame()
    
    def build_object_history(self) -> Dict[int, List[str]]:
        """Build history of frames where each object appears."""
        history = {}
        for object_id in self.results_df['object_id'].unique():
            object_frames = self.results_df[self.results_df['object_id'] == object_id]['name'].tolist()
            history[object_id] = sorted(object_frames)
        return history
    
    def setup_plot(self):
        """Setup the interactive plot with controls."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(26, 12))
        
        # Main image subplot (smaller to make room for controls)
        self.ax_image = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=2)
        
        # Object history subplot (smaller)
        self.ax_history = plt.subplot2grid((3, 6), (2, 0), colspan=3)
        
        # Instructions panel (much wider)
        self.ax_instructions = plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=3)
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # Initial display
        self.update_display()
        
        plt.tight_layout()
        plt.show()
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Frame slider (positioned at the very bottom, no label)
        self.slider_ax = plt.axes((0.65, 0.02, 0.25, 0.03))
        self.slider = Slider(self.slider_ax, '', 0, len(self.all_frames)-1, 
                           valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
    
    def print_instructions(self):
        """Print keyboard shortcuts instructions."""
        print("\n" + "="*60)
        print("INTERACTIVE TRACKING VIEWER - KEYBOARD SHORTCUTS")
        print("="*60)
        print("Navigation:")
        print("  [←] [→]  - Previous/Next frame")
        print("  [Home]   - First frame")
        print("  [End]    - Last frame")
        print("  [Space]  - Next frame")
        print("  [Enter]  - Next frame")
        print()
        print("Other:")
        print("  [q]      - Quit viewer")
        print("  [s]      - Save current view")
        print()
        print("Object Trajectories:")
        print("  - Bottom plot shows object trajectories")
        print("  - Current frame objects are highlighted")
        print("  - Red vertical line shows current frame")
        print("="*60)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'left' or event.key == 'a':
            self.prev_frame()
        elif event.key == 'right' or event.key == 'd':
            self.next_frame()
        elif event.key == 'home':
            self.first_frame()
        elif event.key == 'end':
            self.last_frame()
        elif event.key == 'q':
            plt.close('all')
            print("Viewer closed.")
        elif event.key == 's':
            self.save_current_view()
    
    def save_current_view(self):
        """Save current view as image."""
        filename = f"tracking_view_frame_{self.current_frame_idx:03d}.png"
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"View saved as: {filename}")
    
    def update_display(self):
        """Update the display with current frame."""
        # Clear previous plots
        self.ax_image.clear()
        self.ax_history.clear()
        self.ax_instructions.clear()
        
        # Load and display current frame
        self.display_frame()
        
        # Display object history
        self.display_object_history()
        
        # Display instructions
        self.display_instructions()
        
        # Update frame info
        self.ax_image.set_title(f'Frame {self.current_frame_idx + 1}/{len(self.all_frames)}: {self.current_frame}', 
                               fontsize=14, fontweight='bold')
        
        # Update slider
        self.slider.set_val(self.current_frame_idx)
        
        plt.draw()
    
    def display_frame(self):
        """Display the current frame with tracking results."""
        # Load image
        image_path = os.path.join(self.images_dir, self.current_frame)
        if not os.path.exists(image_path):
            self.ax_image.text(0.5, 0.5, f'Image not found: {self.current_frame}', 
                             ha='center', va='center', transform=self.ax_image.transAxes)
            return
        
        image = cv2.imread(image_path)
        if image is None:
            self.ax_image.text(0.5, 0.5, f'Failed to load image: {self.current_frame}', 
                             ha='center', va='center', transform=self.ax_image.transAxes)
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ax_image.imshow(image_rgb)
        
        # Get tracking results for current frame
        frame_results = self.results_df[self.results_df['name'] == self.current_frame]
        
        # Draw bounding boxes
        for _, result in frame_results.iterrows():
            x_center = float(result['x_center'])
            y_center = float(result['y_center'])
            width = float(result['width'])
            height = float(result['height'])
            object_id = int(result['object_id'])
            label = str(result['label'])
            
            # Calculate corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Choose color
            color = self.colors[object_id % len(self.colors)]
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                   edgecolor=color, facecolor='none')
            self.ax_image.add_patch(rect)
            
            # Add label with object ID
            label_text = f"ID:{object_id} {label}"
            self.ax_image.text(x1, y1-10, label_text, color=color, fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add center point
            self.ax_image.plot(x_center, y_center, 'o', color=color, markersize=6)
        
        self.ax_image.axis('off')
    
    def display_object_history(self):
        """Display object history timeline."""
        # Get current frame objects
        frame_mask = self.results_df['name'] == self.current_frame
        current_objects = self.results_df.loc[frame_mask, 'object_id'].drop_duplicates().tolist()
        
        # Create timeline
        frame_indices = list(range(len(self.all_frames)))
        
        # Plot object trajectories
        for object_id in current_objects:
            if object_id in self.object_history:
                object_frames = self.object_history[object_id]
                object_indices = [self.all_frames.index(frame) for frame in object_frames]
                
                color = self.colors[object_id % len(self.colors)]
                
                # Plot trajectory line
                self.ax_history.plot(object_indices, [object_id] * len(object_indices), 
                                   'o-', color=color, linewidth=2, markersize=6, label=f'Object {object_id}')
                
                # Highlight current frame
                if self.current_frame in object_frames:
                    current_idx = self.all_frames.index(self.current_frame)
                    self.ax_history.plot(current_idx, object_id, 'o', color=color, 
                                       markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Highlight current frame
        self.ax_history.axvline(x=self.current_frame_idx, color='red', linestyle='--', alpha=0.7)
        
        self.ax_history.set_xlabel('Frame Index')
        self.ax_history.set_ylabel('Object ID')
        self.ax_history.set_title('Object Trajectories (Current Frame Objects Highlighted)')
        self.ax_history.grid(True, alpha=0.3)
        self.ax_history.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis limits
        self.ax_history.set_xlim(-1, len(self.all_frames))
    
    def display_instructions(self):
        """Display keyboard shortcuts in the side panel."""
        instructions = [
            "KEYBOARD SHORTCUTS",
            "",
            "Navigation:",
            "  ← →     Previous/Next frame",
            "  A D     Alternative navigation",
            "  Home    First frame",
            "  End     Last frame",
            "",
            "Other:",
            "  Q       Quit viewer",
            "  S       Save current view",
            "",
            f"Current Status:",
            f"  Frame: {self.current_frame_idx + 1}/{len(self.all_frames)}",
            f"  Objects: {len(self.results_df[self.results_df['name'] == self.current_frame])}",
            f"  File: {os.path.basename(self.tracker_file)}",
            "",
            "Object Trajectories:",
            "  Bottom plot shows object paths",
            "  Current frame objects highlighted",
            "  Red line shows current frame",
            "",
            "",
            "",
            "Frame Slider:",
            "  Drag to navigate frames"
        ]
        
        y_pos = 0.95
        for instruction in instructions:
            if instruction.startswith("  "):
                # Sub-item
                self.ax_instructions.text(0.08, y_pos, instruction, fontsize=11, 
                                        transform=self.ax_instructions.transAxes)
            elif instruction == "":
                # Empty line
                pass
            else:
                # Header
                self.ax_instructions.text(0.08, y_pos, instruction, fontsize=13, fontweight='bold',
                                        transform=self.ax_instructions.transAxes)
            y_pos -= 0.035
        
        self.ax_instructions.axis('off')
    
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < len(self.all_frames) - 1:
            self.current_frame_idx += 1
            self.current_frame = self.all_frames[self.current_frame_idx]
            self.update_display()
    
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.current_frame = self.all_frames[self.current_frame_idx]
            self.update_display()
    
    def first_frame(self):
        """Go to first frame."""
        self.current_frame_idx = 0
        self.current_frame = self.all_frames[0]
        self.update_display()
    
    def last_frame(self):
        """Go to last frame."""
        self.current_frame_idx = len(self.all_frames) - 1
        self.current_frame = self.all_frames[-1]
        self.update_display()
    
    def on_slider_change(self, val):
        """Handle slider change."""
        frame_idx = int(val)
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self.current_frame = self.all_frames[frame_idx]
            self.update_display()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive Tracking Viewer")
    parser.add_argument("--tracker_file", required=True,
                       help="Path to tracking results TSV file")
    parser.add_argument("--images_dir", required=True,
                       help="Directory containing images")
    
    args = parser.parse_args()
    
    print("Starting Interactive Tracking Viewer...")
    print(f"Tracker file: {args.tracker_file}")
    print(f"Images directory: {args.images_dir}")
    print("Use keyboard shortcuts to navigate (press 'h' for help)")
    
    # Start the interactive viewer
    viewer = InteractiveTrackingViewer(args.tracker_file, args.images_dir)

if __name__ == "__main__":
    main() 