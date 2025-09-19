#!/usr/bin/env python3
"""
Object Tracking Algorithm for Autobrains Data Engineering Assignment
Author: Roni Levi

This script implements a multi-object tracking algorithm that assigns consistent
object IDs across consecutive frames based on bounding box overlap and distance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
import os


class ObjectTracker:
    """
    Multi-object tracker that maintains consistent object IDs across frames.
    
    Uses IoU (Intersection over Union) and distance-based matching to associate
    detections between consecutive frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, distance_threshold: float = 200.0):
        """
        Initialize the tracker.
        
        Args:
            iou_threshold: Minimum IoU for considering two detections as the same object
            distance_threshold: Maximum distance between centers for matching
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.next_object_id = 1
        self.active_tracks: Dict[int, Dict] = {}
        self.track_history: List[Dict] = []
        self.inactive_tracks: Dict[int, Dict] = {}  # Store recently lost tracks
        self.track_age: Dict[int, int] = {}  # Track how long each track has been missing
        self.max_track_age = 20  # Maximum frames a track can be missing before being deleted
        
    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: Dictionaries with keys 'center_x', 'center_y', 'width', 'height'
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert center coordinates to corner coordinates
        x1_min = box1['center_x'] - box1['width'] / 2
        x1_max = box1['center_x'] + box1['width'] / 2
        y1_min = box1['center_y'] - box1['height'] / 2
        y1_max = box1['center_y'] + box1['height'] / 2
        
        x2_min = box2['center_x'] - box2['width'] / 2
        x2_max = box2['center_x'] + box2['width'] / 2
        y2_min = box2['center_y'] - box2['height'] / 2
        y2_max = box2['center_y'] + box2['height'] / 2
        
        # Calculate intersection
        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        
        if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
            return 0.0
            
        intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
        
        # Calculate union
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def calculate_distance(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate Euclidean distance between centers of two bounding boxes.
        
        Args:
            box1, box2: Dictionaries with keys 'center_x', 'center_y'
            
        Returns:
            Distance between centers
        """
        dx = box1['center_x'] - box2['center_x']
        dy = box1['center_y'] - box2['center_y']
        return np.sqrt(dx * dx + dy * dy)
    
    def find_best_match(self, detection: Dict, candidates: List[Dict]) -> Optional[int]:
        """
        Find the best matching track for a detection.
        
        Args:
            detection: Current detection
            candidates: List of candidate tracks
            
        Returns:
            Object ID of best match, or None if no good match found
        """
        best_match_id = None
        best_score = 0.0
        
        for track_id, track_info in candidates.items():
            # Calculate IoU
            iou = self.calculate_iou(detection, track_info)
            
            # Calculate distance
            distance = self.calculate_distance(detection, track_info)
            
            # More flexible matching criteria for re-identification
            if iou >= self.iou_threshold and distance <= self.distance_threshold:
                # Normalize distance to 0-1 range (closer is better)
                distance_score = max(0, 1 - (distance / self.distance_threshold))
                
                # Label consistency bonus (but don't penalize too heavily for mismatches)
                label_bonus = 0.1 if detection['label'] == track_info['label'] else 0.0
                
                combined_score = 0.7 * iou + 0.2 * distance_score + label_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_id = track_id
        
        return best_match_id
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections for current frame
            
        Returns:
            List of detections with assigned object IDs
        """
        # Create a copy of active tracks for matching
        available_tracks = self.active_tracks.copy()
        matched_detections = []
        
        # First pass: try to match detections with active tracks
        for detection in detections:
            best_match_id = self.find_best_match(detection, available_tracks)
            
            if best_match_id is not None:
                # Assign existing object ID
                detection['object_id'] = best_match_id
                matched_detections.append(detection)
                
                # Update track info and remove from available tracks
                self.active_tracks[best_match_id] = detection
                del available_tracks[best_match_id]
            else:
                # Try to match with inactive tracks (re-identification)
                best_match_id = self.find_best_match(detection, self.inactive_tracks)
                
                if best_match_id is not None:
                    # Re-identify: assign the inactive track's ID
                    detection['object_id'] = best_match_id
                    matched_detections.append(detection)
                    
                    # Move track back to active and remove from inactive
                    self.active_tracks[best_match_id] = detection
                    del self.inactive_tracks[best_match_id]
                    del self.track_age[best_match_id]
                else:
                    # Create new track
                    detection['object_id'] = self.next_object_id
                    matched_detections.append(detection)
                    self.active_tracks[self.next_object_id] = detection
                    self.next_object_id += 1
        
        # Move unmatched active tracks to inactive
        for track_id in available_tracks:
            self.inactive_tracks[track_id] = self.active_tracks[track_id]
            self.track_age[track_id] = 0
            del self.active_tracks[track_id]
        
        # Age inactive tracks
        for track_id in list(self.track_age.keys()):
            self.track_age[track_id] += 1
            if self.track_age[track_id] > self.max_track_age:
                # Remove old inactive tracks
                del self.inactive_tracks[track_id]
                del self.track_age[track_id]
        
        # Post-processing: remove obvious duplicates within the same frame
        matched_detections = self.remove_duplicates_in_frame(matched_detections)
        
        # Store track history
        for detection in matched_detections:
            self.track_history.append(detection.copy())
        
        return matched_detections
    
    def remove_duplicates_in_frame(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate detections within the same frame.
        
        Args:
            detections: List of detections for current frame
            
        Returns:
            List of detections with duplicates removed
        """
        if len(detections) <= 1:
            return detections
        
        # Check for duplicates across all detections, not just within same label
        to_remove = set()
        
        for i in range(len(detections)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(detections)):
                if j in to_remove:
                    continue
                
                # Check if these are likely the same object
                iou = self.calculate_iou(detections[i], detections[j])
                distance = self.calculate_distance(detections[i], detections[j])
                
                # Check if labels are compatible
                label1 = detections[i]['label']
                label2 = detections[j]['label']
                labels_compatible = (label1 == label2 or 
                                  (label1 in ['CAR', 'TRUCK', 'VEHICLE'] and 
                                   label2 in ['CAR', 'TRUCK', 'VEHICLE']) or
                                  (label1 in ['PEDESTRIAN', 'PERSON'] and 
                                   label2 in ['PEDESTRIAN', 'PERSON']))
                
                # If IoU is high or distance is very small, and labels are compatible, they're likely duplicates
                if labels_compatible and (iou > 0.3 or distance < 100):
                    # Keep the detection with larger bounding box (more complete)
                    area_i = detections[i]['width'] * detections[i]['height']
                    area_j = detections[j]['width'] * detections[j]['height']
                    
                    if area_i >= area_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Return non-removed detections
        filtered_detections = []
        for i, detection in enumerate(detections):
            if i not in to_remove:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def cleanup_old_tracks(self, max_age: int = 5):
        """
        Remove tracks that haven't been seen for too long.
        
        Args:
            max_age: Maximum number of frames a track can be missing
        """
        # For simplicity, we'll keep all tracks active
        # In a more sophisticated implementation, we could track frame counts
        pass


def load_detections(file_path: str) -> pd.DataFrame:
    """
    Load detections from TSV file.
    
    Args:
        file_path: Path to the detections TSV file
        
    Returns:
        DataFrame with detections
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Loaded {len(df)} detections from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading detections: {e}")
        raise


def process_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process detections and assign object IDs using tracking algorithm.
    
    Args:
        df: DataFrame with detections
        
    Returns:
        DataFrame with object IDs assigned
    """
    # Initialize tracker
    tracker = ObjectTracker(iou_threshold=0.3, distance_threshold=200.0)
    
    # Group by frame name to process frame by frame
    frames = df.groupby('name')
    all_tracked_detections = []
    
    print(f"Processing {len(frames)} frames...")
    
    for frame_name, frame_detections in frames:
        # Convert frame detections to list of dictionaries
        detections = []
        for _, row in frame_detections.iterrows():
            detection = {
                'name': row['name'],
                'center_x': row['x_center'],
                'center_y': row['y_center'],
                'width': row['width'],
                'height': row['height'],
                'label': row['label']
            }
            detections.append(detection)
        
        # Update tracks
        tracked_detections = tracker.update_tracks(detections)
        all_tracked_detections.extend(tracked_detections)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(all_tracked_detections)
    
    # Rename columns to match expected output format
    result_df = result_df.rename(columns={
        'center_x': 'x_center',
        'center_y': 'y_center',
        'width': 'width',
        'height': 'height',
        'label': 'label',
        'object_id': 'object_id'
    })
    
    # Reorder columns to match expected format
    result_df = result_df[['name', 'x_center', 'y_center', 'width', 'height', 'label', 'object_id']]
    
    print(f"Generated {len(result_df)} tracked detections with {result_df['object_id'].nunique()} unique objects")
    
    # Post-processing: merge tracks that are clearly the same object
    result_df = merge_duplicate_tracks(result_df)
    
    print(f"After merging duplicate tracks: {result_df['object_id'].nunique()} unique objects")
    
    return result_df


def merge_duplicate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process to merge tracks that are clearly the same object.
    
    Args:
        df: DataFrame with tracking results
        
    Returns:
        DataFrame with merged tracks
    """
    # Group by object_id to analyze each track
    tracks = df.groupby('object_id')
    
    # Find tracks to merge using improved logic
    tracks_to_merge = {}
    processed_tracks = set()
    
    # Convert frame names to frame numbers for temporal analysis
    frame_mapping = {}
    unique_frames = sorted(df['name'].unique())
    for i, frame_name in enumerate(unique_frames):
        frame_mapping[frame_name] = i
    
    for track_id, track_data in tracks:
        if track_id in processed_tracks:
            continue
            
        # Sort track data by frame
        track_data = track_data.sort_values('name')
        track_data = track_data.reset_index(drop=True)
        
        # Look for other tracks that might be the same object
        for other_track_id, other_track_data in tracks:
            if other_track_id == track_id or other_track_id in processed_tracks:
                continue
                
            # Sort other track data by frame
            other_track_data = other_track_data.sort_values('name')
            other_track_data = other_track_data.reset_index(drop=True)
            
            # Check if these tracks should be merged using improved logic
            if should_merge_tracks_improved(track_data, other_track_data, frame_mapping):
                # Merge these tracks
                if track_id not in tracks_to_merge:
                    tracks_to_merge[track_id] = []
                tracks_to_merge[track_id].append(other_track_id)
                processed_tracks.add(other_track_id)
        
        processed_tracks.add(track_id)
    
    # Apply the merges
    result_df = df.copy()
    for main_track_id, tracks_to_merge_list in tracks_to_merge.items():
        for track_to_merge in tracks_to_merge_list:
            # Replace all instances of track_to_merge with main_track_id
            result_df.loc[result_df['object_id'] == track_to_merge, 'object_id'] = main_track_id
    
    return result_df


def should_merge_tracks_improved(track1_data: pd.DataFrame, track2_data: pd.DataFrame, frame_mapping: Dict) -> bool:
    """
    Determine if two tracks should be merged using improved logic.
    
    Args:
        track1_data: First track data (sorted by frame)
        track2_data: Second track data (sorted by frame)
        frame_mapping: Mapping from frame names to frame numbers
        
    Returns:
        True if tracks should be merged
    """
    # Get frame numbers for temporal analysis
    track1_frames = [frame_mapping[name] for name in track1_data['name']]
    track2_frames = [frame_mapping[name] for name in track2_data['name']]
    
    track1_start, track1_end = min(track1_frames), max(track1_frames)
    track2_start, track2_end = min(track2_frames), max(track2_frames)
    
    # Check if labels are compatible
    label1 = track1_data['label'].iloc[0]
    label2 = track2_data['label'].iloc[0]
    label_compatible = (label1 == label2 or 
                      (label1 in ['PEDESTRIAN', 'PERSON'] and 
                       label2 in ['PEDESTRIAN', 'PERSON']) or
                      (label1 in ['CAR', 'VEHICLE'] and 
                       label2 in ['CAR', 'VEHICLE']))
    
    if not label_compatible:
        return False
    
    # Case 1: Tracks overlap in time - check for spatial overlap
    if not (track1_end < track2_start or track2_end < track1_start):
        # Tracks overlap temporally, check if they're spatially close
        for _, det1 in track1_data.iterrows():
            for _, det2 in track2_data.iterrows():
                if det1['name'] == det2['name']:  # Same frame
                    distance = np.sqrt((det1['x_center'] - det2['x_center'])**2 + 
                                     (det1['y_center'] - det2['y_center'])**2)
                    iou = calculate_iou_simple(det1, det2)
                    
                    # If they're very close in the same frame, they're likely the same object
                    if distance < 50 or iou > 0.3:
                        return True
    
    # Case 2: Tracks don't overlap - check for temporal continuity
    else:
        # Find the closest detections in time
        if track1_end < track2_start:  # track1 ends before track2 starts
            last_det1 = track1_data.iloc[-1]
            first_det2 = track2_data.iloc[0]
            frame_gap = track2_start - track1_end
        else:  # track2 ends before track1 starts
            last_det2 = track2_data.iloc[-1]
            first_det1 = track1_data.iloc[0]
            frame_gap = track1_start - track2_end
            # Swap for consistent analysis
            last_det1, first_det2 = first_det1, last_det2
        
        # Calculate distance and IoU between closest detections
        distance = np.sqrt((last_det1['x_center'] - first_det2['x_center'])**2 + 
                          (last_det1['y_center'] - first_det2['y_center'])**2)
        iou = calculate_iou_simple(last_det1, first_det2)
        
        # For cars, be more lenient with temporal gaps and movement
        if label1 == 'CAR' and label2 == 'CAR':
            # Allow larger gaps for cars (they can move quickly)
            max_frame_gap = 30
            max_distance = 300
            
            # Check if the movement direction is consistent (car moving left to right or vice versa)
            if len(track1_data) > 1 and len(track2_data) > 1:
                # Calculate movement direction for track1 (last few detections)
                track1_recent = track1_data.tail(min(3, len(track1_data)))
                x_movement1 = track1_recent['x_center'].iloc[-1] - track1_recent['x_center'].iloc[0]
                
                # Calculate movement direction for track2 (first few detections)
                track2_early = track2_data.head(min(3, len(track2_data)))
                x_movement2 = track2_early['x_center'].iloc[-1] - track2_early['x_center'].iloc[0]
                
                # If both tracks are moving in the same direction, be more lenient
                if (x_movement1 * x_movement2) > 0:  # Same direction
                    max_distance = 400
                    max_frame_gap = 50
            
            if frame_gap <= max_frame_gap and distance <= max_distance:
                return True
        
        # For other objects, use stricter criteria
        else:
            if frame_gap <= 10 and distance <= 150 and (iou > 0.1 or distance < 80):
                return True
    
    return False


def should_merge_tracks(detection1: pd.Series, detection2: pd.Series) -> bool:
    """
    Determine if two detections from different tracks should be merged.
    
    Args:
        detection1: Last detection from first track
        detection2: First detection from second track
        
    Returns:
        True if tracks should be merged
    """
    # Calculate distance between detections
    distance = np.sqrt((detection1['x_center'] - detection2['x_center'])**2 + 
                      (detection1['y_center'] - detection2['y_center'])**2)
    
    # Calculate IoU
    iou = calculate_iou_simple(detection1, detection2)
    
    # Check if labels are compatible (same or similar)
    label_compatible = (detection1['label'] == detection2['label'] or 
                      (detection1['label'] in ['PEDESTRIAN', 'PERSON'] and 
                       detection2['label'] in ['PEDESTRIAN', 'PERSON']) or
                      (detection1['label'] in ['CAR', 'VEHICLE'] and 
                       detection2['label'] in ['CAR', 'VEHICLE']))
    
    # Merge if distance is small and IoU is reasonable, or if very close regardless of IoU
    if distance < 100 and (iou > 0.1 or distance < 50):
        return True
    
    # Also merge if very close and labels are compatible
    if distance < 80 and label_compatible:
        return True
        
    return False


def calculate_iou_simple(det1: pd.Series, det2: pd.Series) -> float:
    """
    Calculate IoU between two detections.
    
    Args:
        det1, det2: Detection series with x_center, y_center, width, height
        
    Returns:
        IoU value
    """
    # Convert to corner coordinates
    x1_min = det1['x_center'] - det1['width'] / 2
    x1_max = det1['x_center'] + det1['width'] / 2
    y1_min = det1['y_center'] - det1['height'] / 2
    y1_max = det1['y_center'] + det1['height'] / 2
    
    x2_min = det2['x_center'] - det2['width'] / 2
    x2_max = det2['x_center'] + det2['width'] / 2
    y2_min = det2['y_center'] - det2['height'] / 2
    y2_max = det2['y_center'] + det2['height'] / 2
    
    # Calculate intersection
    intersection_x_min = max(x1_min, x2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_min = max(y1_min, y2_min)
    intersection_y_max = min(y1_max, y2_max)
    
    if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
        return 0.0
        
    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    
    # Calculate union
    area1 = det1['width'] * det1['height']
    area2 = det2['width'] * det2['height']
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def save_results(df: pd.DataFrame, output_path: str):
    """
    Save tracking results to TSV file.
    
    Args:
        df: DataFrame with tracking results
        output_path: Path to save the output file
    """
    try:
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        raise


def main():
    """Main function to run the tracking algorithm."""
    parser = argparse.ArgumentParser(description='Object Tracking Algorithm')
    parser.add_argument('--input', '-i', default='detections.tsv', 
                       help='Input detections TSV file (default: detections.tsv)')
    parser.add_argument('--output', '-o', default='Roni_Roitbord_detections.tsv',
                       help='Output tracked detections TSV file (default: Roni_Roitbord_detections.tsv)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    try:
        # Load detections
        print("Loading detections...")
        df = load_detections(args.input)
        
        # Process detections with tracking
        print("Running tracking algorithm...")
        tracked_df = process_detections(df)
        
        # Save results
        print("Saving results...")
        save_results(tracked_df, args.output)
        
        print("Tracking completed successfully!")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"- Total detections: {len(tracked_df)}")
        print(f"- Unique objects tracked: {tracked_df['object_id'].nunique()}")
        print(f"- Frames processed: {tracked_df['name'].nunique()}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())