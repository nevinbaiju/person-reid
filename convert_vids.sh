#!/bin/bash

# Specify the input and output folders
input_folder="temp/drawn_vids"
output_folder="temp/processed_vids"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Convert all AVI files in the input folder to MP4
for avi_file in "$input_folder"/*.avi; do
    if [ -e "$avi_file" ]; then
        filename=$(basename -- "$avi_file")
        filename_noext="${filename%.*}"
        output_file="$output_folder/$filename_noext.mp4"

        # Run ffmpeg command
        ffmpeg -i "$avi_file" -c:v libx264 -crf 23 -c:a aac -strict experimental -b:a 192k -ac 2 "$output_file"

        echo "Converted: $filename to $filename_noext.mp4"
    fi
done

echo "Conversion complete!"
touch complete
