#!/usr/bin/env bash

EXTENSION=".png"
ASSETS_PATH="data/assets"

case $1 in
"cells")
    find $ASSETS_PATH/cells -name "*$EXTENSION" -type f -delete
    echo "cellss directory cleaned"
    ;;
"out")
    find $ASSETS_PATH/out -name "*$EXTENSION" -type f -delete
    echo "out directory cleaned"
    ;;
"*")
    find $ASSETS_PATH -name "*$EXTENSION" -type f -delete
    echo "directories cleaned"
    ;;
esac






