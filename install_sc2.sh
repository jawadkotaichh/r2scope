#!/bin/bash
# Install SC2 and add the custom maps
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$EXP_DIR" ] && [ -d "$EXP_DIR/RODE" ]; then
        PROJECT_DIR="$EXP_DIR/RODE"
else
        PROJECT_DIR="$SCRIPT_DIR"
fi

download() {
        URL="$1"
        OUTPUT="$2"

        if command -v wget >/dev/null 2>&1; then
                wget -O "$OUTPUT" "$URL"
        elif command -v curl >/dev/null 2>&1; then
                curl -L -o "$OUTPUT" "$URL"
        else
                echo "Neither wget nor curl is installed. Install one of them and rerun this script." >&2
                exit 1
        fi
}

extract_zip() {
        ZIP_FILE="$1"
        PASSWORD="${2:-}"

        if command -v unzip >/dev/null 2>&1; then
                if [ -n "$PASSWORD" ]; then
                        unzip -o -P "$PASSWORD" "$ZIP_FILE"
                else
                        unzip -o "$ZIP_FILE"
                fi
                return
        fi

        if command -v python3 >/dev/null 2>&1; then
                ZIP_FILE="$ZIP_FILE" ZIP_PASSWORD="$PASSWORD" python3 - <<'PY'
import os
import zipfile

zip_file = os.environ["ZIP_FILE"]
password = os.environ.get("ZIP_PASSWORD") or None
pwd = password.encode("utf-8") if password else None

with zipfile.ZipFile(zip_file) as zf:
    zf.extractall(pwd=pwd)
PY
                return
        fi

        echo "Neither unzip nor python3 is available to extract $ZIP_FILE." >&2
        exit 1
}

echo "PROJECT_DIR: $PROJECT_DIR"
cd "$PROJECT_DIR"

mkdir -p 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d "$SC2PATH/Versions" ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        download http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip SC2.4.10.zip
        extract_zip SC2.4.10.zip iagreetotheeula
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d "$MAP_DIR" ]; then
        mkdir -p "$MAP_DIR"
fi

cd ..
download https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip SMAC_Maps.zip
extract_zip SMAC_Maps.zip
mv SMAC_Maps "$MAP_DIR"
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'
