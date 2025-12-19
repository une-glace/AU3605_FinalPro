# 该脚本用于批量解压指定目录下的 .gz 图像文件（如血管标注数据集）
import os
import gzip
import shutil


def decompress_gz_root(root_dir, overwrite=False):
    for current_root, _, files in os.walk(root_dir):
        for name in files:
            if not name.endswith(".gz"):
                continue
            gz_path = os.path.join(current_root, name)
            out_path = os.path.join(current_root, name[:-3])
            if os.path.exists(out_path) and not overwrite:
                print(f"Skip {out_path} (already exists)")
                continue
            print(f"Decompress {gz_path} -> {out_path}")
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "..")
    decompress_gz_root(os.path.abspath(target_dir), overwrite=False)

