# make_residual_panels_movie.py
import os, glob
from PIL import Image, ImageOps
import imageio.v2 as imageio

BASE = os.path.join("visualization", "movie")
YEARS = [1880, 1926, 1957, 1980, 1999]
OUT_FRAMES_DIR = os.path.join(BASE, "_residual_panels")
OUT_MP4 = os.path.join(BASE, "residuals_panels.mp4")

os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

def parse_iter(path):
    # .../residual_yearYYYY_iterationNNN.png  -> NNN (int)
    name = os.path.basename(path)
    # split on "_iteration", then strip ".png"
    try:
        return int(name.split("_iteration")[-1].split(".")[0])
    except Exception:
        return None

# 1) Discover iterations per year (with logging)
iters_by_year = {}
for y in YEARS:
    files = glob.glob(os.path.join(BASE, f"residual_year{y}_iteration*.png"))
    iters = sorted(i for i in (parse_iter(f) for f in files) if i is not None)
    iters_by_year[y] = iters
    if iters:
        print(f"Year {y}: {len(iters)} frames, min={iters[0]}, max={iters[-1]}")
    else:
        print(f"Year {y}: NO frames found under {BASE}")

# 2) Find common iterations across ALL years
common_iters = sorted(set(iters_by_year[YEARS[0]]).intersection(*[set(iters_by_year[y]) for y in YEARS[1:]]))
if not common_iters:
    print("\nNo common iteration indices across all years.")
    print("Double-check you are running from the project root so BASE='visualization/movie' resolves correctly.")
    # You can bail here, or continue with a relaxed strategy; we’ll bail to avoid mixing years.
    raise SystemExit(1)

print(f"\nBuilding panels for {len(common_iters)} iterations. First few: {common_iters[:10]}")

# 3) Build and save panel frames
gap = 8  # px between images
saved_frames = []
for i in common_iters:
    imgs = []
    missing = []
    for y in YEARS:
        fn = os.path.join(BASE, f"residual_year{y}_iteration{i}.png")
        if not os.path.exists(fn):
            missing.append(fn)
            break
        imgs.append(Image.open(fn).convert("RGBA"))
    if missing:
        print(f"[skip] iteration {i}: missing {missing[0]}")
        continue

    # Make same height & stack horizontally with a small gap
    H = max(im.height for im in imgs)
    imgs = [ImageOps.contain(im, (int(im.width * H / im.height), H), method=Image.BICUBIC) for im in imgs]
    total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    panel = Image.new("RGBA", (total_w, H), (255, 255, 255, 0))

    x = 0
    for k, im in enumerate(imgs):
        panel.paste(im, (x, 0), im)
        if k < len(imgs) - 1:
            x += im.width + gap

    out_path = os.path.join(OUT_FRAMES_DIR, f"frame_{i:04d}.png")
    panel.convert("RGB").save(out_path)
    saved_frames.append(out_path)

print(f"Saved {len(saved_frames)} panel frames to {OUT_FRAMES_DIR}")

if not saved_frames:
    raise SystemExit("No panel frames were created; aborting movie write.")

# 4) Write MP4 with imageio (no ffmpeg dependency)
# Adjust fps to taste
fps = 3
print(f"Writing {OUT_MP4} at {fps} fps …")
with imageio.get_writer(OUT_MP4, fps=fps) as writer:
    for f in saved_frames:
        writer.append_data(imageio.imread(f))
print("Done.")
