#!/bin/bash

# Set temporary filename for the tar file
TEMP_TAR="opus_data_temp.tar.gz"

# Create tar file with files listed in tar_list.txt
if [ ! -f tar_list.txt ]; then
    echo "Error: tar_list.txt not found"
    exit 1
fi

# First create just the temp tar file
tar cfvz "$TEMP_TAR" -T tar_list.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to create tar file"
    rm -f "$TEMP_TAR" 2>/dev/null
    exit 1
fi

# Calculate SHA256 hash of the tar file
SHA256=$(sha256sum "$TEMP_TAR" | cut -d' ' -f1)


# Rename the file with the hash
FINAL_NAME="opus_data-${SHA256}.tar.gz"
mv "$TEMP_TAR" "$FINAL_NAME"

# Clean up the temporary tar file
rm -f "$TEMP_TAR"

echo "Successfully created $FINAL_NAME"
echo "SHA256: $SHA256"