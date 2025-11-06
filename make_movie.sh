#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (tweak as needed)
# -----------------------------

# Frame rate for the final videos
FRAMERATE_MP4=1
FRAMERATE_GIF=2

# Text overlay settings
FONTSIZE=48
FONTCOLOR="white"
BOX=1
BOXCOLOR="black@0.5"
BOXBORDERW=12

# If your ffmpeg can't find a font automatically, set FONTFILE to a .ttf
FONTFILE="${FONTFILE:-}"

# Text position: centered near the top
# (Examples: "x=w-tw-20:y=h-th-20" for bottom-right; "x=20:y=20" for top-left)
TEXT_POS="x=(w-tw)/2:y=20"

# Output filenames
OUT_MP4="extent_temp_stacked.mp4"
OUT_GIF="extent_temp_stacked.gif"

# Temporary frames directory
FRAMES_DIR="frames"

# For H.264: ensure even width/height without resampling (pads 1px if needed)
EVENIZE_PAD="pad=ceil(iw/2)*2:ceil(ih/2)*2"

# -----------------------------
# Checks
# -----------------------------

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found in PATH." >&2
  exit 1
}

shopt -s nullglob
ext_imgs=(extent_*.png)
tmp_imgs=(temp_*.png)
if [ ${#ext_imgs[@]} -eq 0 ] || [ ${#tmp_imgs[@]} -eq 0 ]; then
  echo "No extent_*.png or temp_*.png images found in the current directory." >&2
  exit 1
fi

# -----------------------------
# Build list of common indices
# -----------------------------

readarray -t COMMON <<<"$(
  comm -12 \
    <(printf '%s\n' "${ext_imgs[@]}" | sed -E 's/.*_([0-9]+)\.png/\1/' | sort -n | uniq) \
    <(printf '%s\n' "${tmp_imgs[@]}" | sed -E 's/.*_([0-9]+)\.png/\1/' | sort -n | uniq)
)"

if [ ${#COMMON[@]} -eq 0 ]; then
  echo "No matching extent/temp indices found." >&2
  exit 1
fi

echo "Found ${#COMMON[@]} matching indices."

# -----------------------------
# Prepare frames folder
# -----------------------------

rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"

# -----------------------------
# Generate stacked frames
# -----------------------------

i=0
for n in "${COMMON[@]}"; do
  printf -v seq "%04d" "$i"
  in_ext="extent_${n}.png"
  in_tmp="temp_${n}.png"
  out_png="${FRAMES_DIR}/frame_${seq}.png"

  # Choose drawtext invocation depending on FONTFILE presence
  if [ -n "$FONTFILE" ]; then
    DRAWTEXT="drawtext=fontfile='${FONTFILE}':text='${n}':fontsize=${FONTSIZE}:fontcolor=${FONTCOLOR}:box=${BOX}:boxcolor=${BOXCOLOR}:boxborderw=${BOXBORDERW}:${TEXT_POS}"
  else
    DRAWTEXT="drawtext=text='${n}':fontsize=${FONTSIZE}:fontcolor=${FONTCOLOR}:box=${BOX}:boxcolor=${BOXCOLOR}:boxborderw=${BOXBORDERW}:${TEXT_POS}"
  fi

  # Scale temp to match extent using scale2ref, then vstack (extent on top), then overlay text
  ffmpeg -y -loglevel error \
    -i "$in_ext" -i "$in_tmp" \
    -filter_complex "
      [0:v]format=rgba,setsar=1[xt];
      [1:v][xt]scale2ref=w=iw:h=ih:flags=bicubic[bt][xts];
      [xts][bt]vstack=inputs=2,format=rgba,${DRAWTEXT}
    " \
    -frames:v 1 "$out_png"

  ((i+=1))
done

# -----------------------------
# Make MP4 (ensure even dimensions for H.264 with tiny padding)
# -----------------------------
ffmpeg -y -loglevel error -framerate "$FRAMERATE_MP4" -i "${FRAMES_DIR}/frame_%04d.png" \
  -vf "${EVENIZE_PAD}" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$OUT_MP4"

# -----------------------------
# Make GIF (no even-dimension requirement; palette for quality)
# -----------------------------
ffmpeg -y -loglevel error -i "${FRAMES_DIR}/frame_%04d.png" \
  -vf "fps=${FRAMERATE_GIF},split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=new=1" \
  "$OUT_GIF"

echo "Done."
echo "  MP4: ${OUT_MP4}"
echo "  GIF: ${OUT_GIF}"
