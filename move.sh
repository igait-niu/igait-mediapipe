#!/bin/bash
# Directory containing the videos and landmarks
VIDEO_DIR="./"       # adjust to your video directory
LANDMARK_DIR="./training_data" # adjust to your landmark directory
OUTPUT_DIR="./subjects"    # where subject folders will be created

mkdir -p "$OUTPUT_DIR"

# Extract unique subject numbers from video files
for f in "$VIDEO_DIR"/*.mp4; do
    basename "$f" | grep -oP '^\d+'
done | sort -u | while read -r subject; do
    echo "Processing subject: $subject"
    
    mkdir -p "$OUTPUT_DIR/$subject"

    # Move front and side pose videos
    for vid in "$VIDEO_DIR"/${subject}_Front_pose.mp4 "$VIDEO_DIR"/${subject}_Side_pose.mp4; do
        [ -f "$vid" ] && mv "$vid" "$OUTPUT_DIR/$subject/"
    done

    # Move front and side landmark JSON files
    for lm in "$LANDMARK_DIR"/${subject}_Front_landmarks.json "$LANDMARK_DIR"/${subject}_Side_landmarks.json; do
        [ -f "$lm" ] && mv "$lm" "$OUTPUT_DIR/$subject/"
    done
done

echo "Done. Organized into $OUTPUT_DIR/"