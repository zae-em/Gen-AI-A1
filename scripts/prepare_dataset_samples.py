"""Export original dataset pixels embedded in the imported evaluation figures."""
import csv
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "imports/genai-assignment-1/artifacts/results/test"
OUTPUT = ROOT / "sample-images"


def extract(figure_name, row, column, output_path):
    # draw_grid in the original evaluate.py pastes 128px images without scaling.
    x, y = column * 128, 52 + row * 164 + 24
    with Image.open(SOURCE / figure_name) as figure:
        image = figure.convert("RGB").crop((x, y, x + 128, y + 128))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image


def main():
    records, thumbnails = [], []
    pet_rows = json.loads((SOURCE / "universal_representative.json").read_text())[:6]
    for index, row in enumerate(pet_rows):
        figure = f"universal_representative_{index // 4 + 1}.png"
        filename = f"pets/{row['id']}.png"
        thumbnails.append(extract(figure, index % 4, 0, OUTPUT / filename))
        records.append({"file": filename, "id": row["id"], "dataset": "Oxford-IIIT Pet",
                        "split": "test", "source_figure": figure,
                        "row": index % 4, "column": "Clean target"})

    with (SOURCE / "face_per_image.csv").open(newline="") as stream:
        face_rows = list(csv.DictReader(stream))
    counts = [0, 0, 0]
    representative = []
    for row in face_rows:
        style = int(row["style"])
        if counts[style] < 4:
            representative.append(row)
            counts[style] += 1
    selected = [0, 1] + [next(i for i, row in enumerate(representative)
                             if int(row["style"]) == style) for style in (0, 2)]
    for index in selected:
        row = representative[index]
        figure = f"face_representative_{index // 4 + 1}.png"
        filename = f"faces/{row['id']}.png"
        thumbnails.append(extract(figure, index % 4, 1, OUTPUT / filename))
        records.append({"file": filename, "id": row["id"], "dataset": "FS2K",
                        "split": "test", "reference_style": int(row["style"]) + 1,
                        "source_figure": figure, "row": index % 4, "column": "Input photograph"})

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else font
    sheet = Image.new("RGB", (1160, 586), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), "10 dataset samples: 6 pets + 4 face photos", fill="#16222f", font=title_font)
    for index, (record, thumbnail) in enumerate(zip(records, thumbnails)):
        x, y = 20 + (index % 5) * 228, 64 + (index // 5) * 256
        sheet.paste(thumbnail.resize((208, 208), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + 214), record["id"], fill="#16222f", font=font)
        draw.text((x, y + 234), record["dataset"], fill="#516172", font=font)
    sheet.save(OUTPUT / "preview.png")
    metadata = {"source_repository": "https://github.com/AbdullahMasood73/genai-assignment-1",
                "source_commit": "4e841d6f966dcfcc64cd04019619609c855df7fa",
                "image_size": [128, 128], "images": records,
                "extraction": "Original 128x128 dataset pixels cropped from lossless report figures; no model-generated outputs included.",
                "datasets": {"Oxford-IIIT Pet": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                             "FS2K": "https://github.com/DengPingFan/FS2K"}}
    (OUTPUT / "sources.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# 10 dataset samples\n\n"
        "These genuine test-dataset examples were extracted at their original 128 x 128 "
        "resolution from the imported assignment's lossless evaluation figures.\n\n"
        "- pets/: Six Oxford-IIIT Pet images for Universal, Hard-Routed, and Soft MoE restoration. "
        "Choose salt-and-pepper, blur, or occlusion in the app to apply damage.\n"
        "- faces/: Four FS2K photographs for Face-to-Sketch. Choose any of the three styles.\n"
        "- preview.png: A labeled overview of all ten images.\n"
        "- sources.json: Dataset IDs, test split, figure locations, and source attribution.\n\n"
        "Upload an individual PNG from pets/ or faces/ at http://localhost:3000.\n"
        "The report figures embed dataset inputs directly; restored/generated outputs "
        "were excluded from these sample files.\n")
    with zipfile.ZipFile(ROOT / "sample-images.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(OUTPUT.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(ROOT))
    assert len(records) == len({record["id"] for record in records}) == 10
    for record in records:
        with Image.open(OUTPUT / record["file"]) as image:
            assert image.size == (128, 128)
    print("Prepared 10 dataset images, preview.png, sources.json, and sample-images.zip.")


if __name__ == "__main__":
    main()
