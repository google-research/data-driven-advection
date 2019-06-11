#!/bin/bash
# Run this script inside Docker container

# Install this packgae
pip install -e .  # track source code change

# compile protobuf
SRC_DIR=./pde_superresolution_2d
protoc -I=$SRC_DIR --python_out=$SRC_DIR $SRC_DIR/metadata.proto

# Run tests
# TODO: use Bazel to run all tests instead of manually discovering tests
test_files=$(find . -name '*_test.py')
log_file=testing_$(date '+%Y-%m-%d_%H:%M:%S').log
for file in $test_files; do
    # echo $file
    python $file 2>&1 | tee -a $log_file
done

