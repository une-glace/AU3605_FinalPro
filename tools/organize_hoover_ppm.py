# 该脚本用于整理 Adam Hoover 提供的 ppm 图像和标注到 images/ 与 labels/ 目录
import os
import shutil


def organize_hoover(base_dir):
    labels_src = os.path.join(
        base_dir, "Hand labeled vessel network provided by Adam Hoover"
    )
    images_src = os.path.join(
        base_dir, "The twenty images are available packaged in a single archive file"
    )
    labels_dst = os.path.join(base_dir, "labels")
    images_dst = os.path.join(base_dir, "images")

    os.makedirs(labels_dst, exist_ok=True)
    os.makedirs(images_dst, exist_ok=True)

    if os.path.exists(labels_src):
        for name in os.listdir(labels_src):
            if name.endswith(".ppm") and not name.endswith(".gz"):
                src = os.path.join(labels_src, name)
                dst = os.path.join(labels_dst, name)
                shutil.copy2(src, dst)
                print(f"Copy {src} -> {dst}")

    if os.path.exists(images_src):
        for name in os.listdir(images_src):
            if name.endswith(".ppm") and not name.endswith(".gz"):
                src = os.path.join(images_src, name)
                dst = os.path.join(images_dst, name)
                shutil.copy2(src, dst)
                print(f"Copy {src} -> {dst}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hoover_root = os.path.join(base_dir, "..", "大作业二-血管标注")
    organize_hoover(os.path.abspath(hoover_root))

