import os

# --- 配置 ---
# 输出的 TXT 文件名
output_filename = 'combined_scripts.txt'

def combine_python_files():
    """
    将脚本所在目录及其所有子目录中除自身以外的所有.py文件合并到一个TXT文件中。
    """
    try:
        # 获取当前脚本的绝对路径，以便在遍历时排除自己
        # os.path.realpath确保我们获取的是真实的物理路径，避免符号链接等问题
        script_path = os.path.realpath(__file__)
    except NameError:
        # 在某些交互式环境（如Jupyter）中 __file__ 可能未定义
        # 在这种情况下，我们假定脚本在当前工作目录中，并给出一个默认名称
        # 如果您在非标准环境下运行，可能需要手动设置脚本路径
        script_path = os.path.join(os.getcwd(), 'your_script_name.py')
        print(f"警告: 无法通过 __file__ 确定脚本路径。假设脚本路径为: {script_path}")


    # 获取脚本所在的目录
    script_dir = os.path.dirname(script_path)

    # 创建一个列表，用于存储所有符合条件的.py文件的路径
    python_files_to_combine = []

    print("开始扫描目录...")

    # os.walk() 会遍历指定目录及其所有子目录
    # root 是当前正在遍历的文件夹路径
    # dirs 是该文件夹下的子文件夹列表
    # files 是该文件夹下的文件列表
    for root, _, files in os.walk(script_dir):
        for file in files:
            # 检查文件是否以 .py 结尾
            if file.endswith('.py'):
                # 构建文件的完整绝对路径
                file_path = os.path.realpath(os.path.join(root, file))

                # 检查这个文件是否是当前运行的脚本
                if file_path != script_path:
                    python_files_to_combine.append(file_path)
                    print(f"  找到文件: {file_path}")

    # 对文件列表进行排序，以确保每次合并的顺序都一样
    python_files_to_combine.sort()

    if not python_files_to_combine:
        print("没有找到其他需要合并的 Python 文件。")
        return

    print(f"\n共找到 {len(python_files_to_combine)} 个 .py 文件。准备写入到 {output_filename}...")

    try:
        # 打开（或创建）输出文件，使用写入模式'w'和utf-8编码
        with open(os.path.join(script_dir, output_filename), 'w', encoding='utf-8') as outfile:
            # 遍历排序后的文件列表
            for filepath in python_files_to_combine:
                # 写入一个分隔符，标明新文件的开始
                outfile.write(f"\n\n{'='*30}\n")
                outfile.write(f"文件路径: {filepath}\n")
                outfile.write(f"{'='*30}\n\n")

                try:
                    # 打开一个.py文件，使用读取模式'r'和utf-8编码
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        # 读取文件内容并写入到输出文件中
                        outfile.write(infile.read())
                except Exception as e:
                    # 如果读取某个文件时出错，打印错误信息并继续
                    error_message = f"!!! 读取文件失败: {filepath} - {e} !!!\n"
                    print(error_message)
                    outfile.write(error_message)

        print(f"\n成功！所有文件已合并到: {os.path.join(script_dir, output_filename)}")

    except IOError as e:
        print(f"\n错误：无法写入到文件 {output_filename}。原因: {e}")
    except Exception as e:
        print(f"\n处理过程中发生未知错误: {e}")


if __name__ == '__main__':
    combine_python_files()