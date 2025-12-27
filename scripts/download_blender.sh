#!/bin/bash

TARGET_DIR="./objaverse_eval"
BLENDER_VERSION="3.3.21"
BLENDER_URL="https://download.blender.org/release/Blender3.3/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
ARCHIVE_NAME="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
FULL_ARCHIVE_PATH="${TARGET_DIR}/${ARCHIVE_NAME}"

mkdir -p "$TARGET_DIR"

echo "Downloading Blender ${BLENDER_VERSION} into ${TARGET_DIR}..."
wget -c -P "$TARGET_DIR" "$BLENDER_URL"

if [ -f "$FULL_ARCHIVE_PATH" ]; then
    echo "Extracting..."

    tar -xf "$FULL_ARCHIVE_PATH" -C "$TARGET_DIR"
    
    echo "Cleaning up..."
    rm "$FULL_ARCHIVE_PATH"
    
    echo "Done! Blender is in ${TARGET_DIR}/blender-${BLENDER_VERSION}-linux-x64"
else
    echo "Download failed."
    exit 1
fi
