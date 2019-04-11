#!/usr/bin/env bash



cd $1
mkdir -p assets
cd assets

find . -delete # Delete everything (recursively)

#Recreate directory structure
mkdir cellule input out

folders=("eosinofili" "epiteliali" "linfociti" "mastcellule" "mucipare" "neutrofili" "others")
for folder in "${folders[@]}"; do
    mkdir out/$folder
done

echo assets directory is clean and initialized!