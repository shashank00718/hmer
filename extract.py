import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Define paths
TRAIN_FOLDER = r"C:\Users\shash\PycharmProjects\hmer\mathwriting-2024-excerpt\train"
OUTPUT_IMAGE_FOLDER = os.path.join(TRAIN_FOLDER, "extracted_images")
OUTPUT_LABEL_FOLDER = os.path.join(TRAIN_FOLDER, "labels")

# Ensure output folders exist
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def extract_namespace(root):
    """Extracts the namespace dynamically from an XML root element."""
    match = root.tag.rfind("}")
    return root.tag[:match + 1] if match != -1 else ""


def read_inkml_file(filename):
    """Reads an InkML file and extracts strokes and annotations."""
    with open(filename, "r") as f:
        root = ET.fromstring(f.read())

    namespace = extract_namespace(root)
    strokes = []
    label = "unknown"

    for element in root:
        tag_name = element.tag.replace(namespace, "")  # Remove namespace
        if tag_name == 'annotation' and element.attrib.get('type') == 'label':
            label = element.text.strip()
        elif tag_name == 'trace':
            points = element.text.strip().split(',')
            stroke_x, stroke_y = [], []
            for point in points:
                values = point.strip().split(' ')
                if len(values) >= 2:  # Ensure x, y are present
                    x, y = map(float, values[:2])
                    stroke_x.append(x)
                    stroke_y.append(y)
            strokes.append(np.array((stroke_x, stroke_y)))

    return strokes, label


def save_image(strokes, filename):
    """Saves the strokes as a 128 X 128 JPG image without grid, title, or axes."""
    plt.figure(figsize=(5.12, 5.12))  # Scales to 64x64 pixels with dpi=25
    for stroke in strokes:
        plt.plot(stroke[0], stroke[1], linewidth=2, color="black")

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=25)  # Ensures 64x64 output
    plt.close()


def save_label(label, filename):
    """Saves the label in a text file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(label)


def process_inkml_files():
    """Processes all InkML files and saves images and labels."""
    for file in os.listdir(TRAIN_FOLDER):
        if file.endswith(".inkml"):
            file_path = os.path.join(TRAIN_FOLDER, file)
            strokes, label = read_inkml_file(file_path)

            # Save image with original filename
            image_filename = os.path.join(OUTPUT_IMAGE_FOLDER, f"{os.path.splitext(file)[0]}.jpg")
            save_image(strokes, image_filename)

            # Save label with original filename
            label_filename = os.path.join(OUTPUT_LABEL_FOLDER, f"{os.path.splitext(file)[0]}.txt")
            save_label(label, label_filename)

            print(f"Processed: {file} -> {os.path.splitext(file)[0]}.jpg and .txt")


if __name__ == "__main__":
    process_inkml_files()
    print(f"All images saved in {OUTPUT_IMAGE_FOLDER}")
    print(f"All labels saved in {OUTPUT_LABEL_FOLDER}")
