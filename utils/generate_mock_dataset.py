import os
import numpy as np
from PIL import Image

CLASSES = [
    "0_severe_depletion",  # Arid, dry soil, parched land, zero surface water
    "1_moderate_stress",   # Mixed dryland, stressed crops, receding margins
    "2_safe_abundant"      # Surface water bodies, center-pivot crops, dense vegetation
]

TERRAIN_TYPES = {
    0: ["parched_clay_pan", "dried_riverbed", "dune_encroachment", "arid_fallow_field", "thermal_drought_patch"],
    1: ["patchy_dryland_crops", "receding_canal_margin", "semi_arid_shrubland", "declining_moisture_basin", "stressed_orchard"],
    2: ["center_pivot_irrigation", "deep_freshwater_reservoir", "wetland_floodplain", "dense_riparian_forest", "active_canal_network"]
}

def generate_diverse_satellite_patch(class_idx, subtype_idx, img_size=(224, 224)):
    """
    Generates realistic, physically-inspired synthetic satellite textures
    representing different groundwater hydrologic features.
    """
    H, W = img_size
    yy, xx = np.mgrid[:H, :W]
    arr = np.zeros((H, W, 3), dtype=np.float32)

    subtype = TERRAIN_TYPES[class_idx][subtype_idx % len(TERRAIN_TYPES[class_idx])]

    if class_idx == 0:
        # --- SEVERE DEPLETION (High Reflectance, Tan, Red-Brown, Dry Soil) ---
        if subtype == "parched_clay_pan":
            base = np.array([205, 175, 130])
            noise = np.random.normal(0, 15, (H, W, 3))
            arr = base + noise
        elif subtype == "dried_riverbed":
            base = np.array([195, 160, 115])
            arr = base + np.random.normal(0, 12, (H, W, 3))
            # Sinuous dry riverbed with bright gravel/silt
            sin_path = (H // 2) + 25 * np.sin(xx * 0.04)
            bed_mask = np.abs(yy - sin_path) < 14
            arr[bed_mask] = [225, 210, 185]
        elif subtype == "dune_encroachment":
            base = np.array([215, 180, 120])
            stripes = 15 * np.sin((xx + yy) * 0.08)
            arr = base + stripes[..., None] + np.random.normal(0, 10, (H, W, 3))
        elif subtype == "arid_fallow_field":
            base = np.array([180, 150, 105])
            # Grid of parched empty farm parcels
            grid = (((xx // 32) % 2) ^ ((yy // 32) % 2)) * 18
            arr = base + grid[..., None] + np.random.normal(0, 12, (H, W, 3))
        else: # thermal_drought_patch
            base = np.array([190, 135, 95])
            arr = base + np.random.normal(0, 18, (H, W, 3))

    elif class_idx == 1:
        # --- MODERATE STRESS (Patchy Yellow-Green, Sparse Vegetation, Low Moisture) ---
        if subtype == "patchy_dryland_crops":
            base = np.array([140, 155, 85])
            patch = (np.sin(xx * 0.06) * np.cos(yy * 0.06) > 0) * 30
            arr = base + patch[..., None] + np.random.normal(0, 20, (H, W, 3))
        elif subtype == "receding_canal_margin":
            base = np.array([130, 140, 90])
            arr = base + np.random.normal(0, 15, (H, W, 3))
            # Thin muddy receding canal
            canal = np.abs(yy - (H // 2)) < 8
            arr[canal] = [90, 110, 115]
        elif subtype == "semi_arid_shrubland":
            base = np.array([145, 150, 100])
            speckles = (np.random.rand(H, W) > 0.85) * 40
            arr = base - speckles[..., None] + np.random.normal(0, 15, (H, W, 3))
        elif subtype == "declining_moisture_basin":
            base = np.array([125, 135, 80])
            grad = (xx / W) * 35
            arr = base + grad[..., None] + np.random.normal(0, 15, (H, W, 3))
        else: # stressed_orchard
            base = np.array([135, 145, 95])
            rows = ((yy % 16) < 4) * 25
            arr = base - rows[..., None] + np.random.normal(0, 14, (H, W, 3))

    else:
        # --- SAFE & ABUNDANT (Deep Chlorophyll Greens, Dark Surface Water) ---
        if subtype == "center_pivot_irrigation":
            base = np.array([80, 120, 70])
            arr = base + np.random.normal(0, 15, (H, W, 3))
            # Circular vibrant green center-pivot field
            circle = ((xx - W//2)**2 + (yy - H//2)**2) < (W//2.8)**2
            arr[circle] = [35, 175, 55] + np.random.normal(0, 12, (arr[circle].shape[0], 3))
        elif subtype == "deep_freshwater_reservoir":
            base = np.array([55, 130, 65])
            arr = base + np.random.normal(0, 15, (H, W, 3))
            # Irregular deep water body
            water = ((xx - W//2)**2 / 1.5 + (yy - H//2)**2 * 1.2) < (W//3)**2
            arr[water] = [25, 75, 175] + np.random.normal(0, 8, (arr[water].shape[0], 3))
        elif subtype == "wetland_floodplain":
            base = np.array([45, 140, 75])
            water_veins = np.abs(np.sin(xx * 0.05) - (yy / H)) < 0.08
            arr = base + np.random.normal(0, 16, (H, W, 3))
            arr[water_veins] = [30, 95, 185]
        elif subtype == "dense_riparian_forest":
            base = np.array([30, 135, 45])
            texture = np.random.normal(0, 22, (H, W, 3))
            arr = base + texture
        else: # active_canal_network
            base = np.array([40, 145, 55])
            arr = base + np.random.normal(0, 15, (H, W, 3))
            cross_canals = (np.abs(xx - W//2) < 9) | (np.abs(yy - H//2) < 9)
            arr[cross_canals] = [20, 85, 195]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), subtype

def setup_mock_dataset(root_dir="data", samples_per_class={"train": 100, "val": 25, "test": 25}):
    """
    Populates an expanded multi-terrain dataset with descriptive filenames.
    """
    print(f"Generating expanded multi-terrain dataset under '{root_dir}/'...")
    total_imgs = 0
    for split, count in samples_per_class.items():
        for class_idx, class_name in enumerate(CLASSES):
            dir_path = os.path.join(root_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(count):
                img, subtype = generate_diverse_satellite_patch(class_idx, i)
                fname = f"{subtype}_{i:03d}.png"
                img.save(os.path.join(dir_path, fname))
                total_imgs += 1

    print(f"Successfully generated {total_imgs} diverse satellite images across {len(CLASSES)} classes!")

if __name__ == "__main__":
    setup_mock_dataset()
