import os

def rename_audio_files(directory, start=601, end=1000, extension=".wav"):
    """
    将指定目录中的音频文件重新命名为从 start 到 end 的编号。

    参数:
        directory (str): 包含音频文件的目录路径。
        start (int): 起始编号，默认为 1。
        end (int): 结束编号，默认为 600。
        extension (str): 音频文件的扩展名，默认为 ".wav"。
    """
    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()  # 按文件名排序，确保顺序正确

    # 检查文件数量是否超过范围
    if len(files) > end - start + 1:
        print(f"警告：目录中有 {len(files)} 个文件，但指定的编号范围只有 {end - start + 1} 个。")
        print("程序已停止，避免文件丢失。")
        return

    # 重命名文件
    for i, file in enumerate(files):
        new_name = f"sound_{i + start:03d}{extension}"  # 格式化为三位数字，例如 "001.wav"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)  # 重命名文件
        print(f"文件 {file} 已重命名为 {new_name}")

    print("文件重命名完成！")

if __name__ == "__main__":
    # 直接设置目标目录
    audio_directory = r"E:\finalproject\Environment\AudioClassification-Pytorch\dataset_myproject\Trafic\audio\fold4"
    rename_audio_files(audio_directory)