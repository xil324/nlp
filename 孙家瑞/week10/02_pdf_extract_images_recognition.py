"""
作业2: 使用云端Qwen-VL大模型识别PDF中的所有嵌入图片
输入: week04/courseware/2017 Attention Is All You Need.pdf
功能: 提取PDF所有页面中嵌入的所有图片，调用Qwen-VL云端API识别每张图片内容
参考: week10/code/案例-多模态文档解析.ipynb, week10/code/案例-多模态内容理解.ipynb
"""

import os
import fitz  # PyMuPDF，来自案例-多模态文档解析
from dashscope import MultiModalConversation  # 来自week01/code/09_Qwen多模态.py


def extract_all_images_from_pdf(pdf_path: str, output_dir: str = ".") -> list:
    """
    提取PDF所有页面中嵌入的所有图片并保存为文件
    参考: week10/code/案例-多模态文档解析.ipynb

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出图片保存目录

    Returns:
        列表，每个元素是(页码, 图片索引, 保存路径)
    """
    # 打开PDF文档
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"正在处理PDF: {pdf_path}")
    print(f"PDF总页数: {total_pages}")

    all_saved_images = []
    total_images_found = 0

    # 遍历所有页面
    for page_num in range(total_pages):
        page = doc[page_num]
        # 获取页面中所有图片
        image_list = page.get_images(full=True)

        if image_list:
            print(f"第 {page_num + 1} 页发现 {len(image_list)} 张嵌入图片")
            total_images_found += len(image_list)

            # 遍历提取每张图片
            for img_index, img in enumerate(image_list, 1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 构造输出文件名，包含页码和图片索引
                output_filename = f"page{page_num + 1}_img{img_index}.{image_ext}"
                output_path = os.path.join(output_dir, output_filename)

                # 保存图片
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                all_saved_images.append((page_num + 1, img_index, output_path))
                print(f"  已保存: {output_path}")
        else:
            print(f"第 {page_num + 1} 页没有发现嵌入图片")

    doc.close()
    print(f"\n提取完成，共发现并保存 {total_images_found} 张图片")
    return all_saved_images


def parse_image_with_qwen_vl(image_path: str, prompt: str) -> str:
    """
    使用阿里云云端Qwen-VL API解析图片内容
    参考:
    - week01/code/09_Qwen多模态.py
    - week10/code/案例-多模态内容理解.ipynb

    Args:
        image_path: 图片文件路径
        prompt: 提示词

    Returns:
        Qwen-VL解析出的文本内容
    """
    # 获取图片绝对路径，DashScope需要file://格式
    abs_image_path = os.path.abspath(image_path)
    image_url = f"file://{abs_image_path}"

    # 构造多模态对话消息
    messages = [
        {
            "role": "system",
            "content": [
                {"text": "你是一个专业的视觉内容识别助手，能够准确描述图片中的内容、图表、示意图、文字等信息。"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": prompt}
            ]
        }
    ]

    # 从环境变量读取API Key，如果没有则使用占位符提示用户
    api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
    if api_key == "YOUR_API_KEY":
        print("⚠️  警告: 请设置环境变量 DASHSCOPE_API_KEY 或替换代码中的API Key")

    # 调用Qwen-VL云端API
    print(f"  正在调用Qwen-VL API识别图片...")
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-vl-max-latest",
        messages=messages
    )

    # 修复类型提示问题：先检查响应类型再提取
    if hasattr(response, "get"):
        parsed_text = response["output"]["choices"][0]["message"]["content"][0]["text"]
    else:
        # 如果是已经解析好的对象，使用点访问
        parsed_text = response.output.choices[0].message.content[0].text

    return parsed_text


def main():
    # ========== 配置 ==========
    # 输入PDF路径（相对当前脚本位置）
    pdf_path = "../../week04/courseware/2017 Attention Is All You Need.pdf"
    # 输出目录（当前目录）
    output_dir = "."

    # 识别提示词 - 让Qwen-VL描述图片内容
    prompt = """请详细描述这张图片中的内容。这是一篇学术论文《Attention Is All You Need》中的图表或示意图，请详细说明它展示了什么信息，包含哪些结构、数据或文字。"""

    # ========== 执行 ==========
    # 步骤1: 提取PDF所有页面中的所有嵌入图片
    all_images = extract_all_images_from_pdf(os.path.abspath(pdf_path), output_dir)

    if not all_images:
        print("没有图片可识别，程序退出")
        return

    # ========== 步骤2: 对每个图片使用Qwen-VL识别内容 ==========
    all_results = []

    for page_num, img_idx, image_path in all_images:
        print(f"\n----- 第 {page_num} 页 / 图片 {img_idx}: {image_path} -----")

        # 识别图片内容
        result = parse_image_with_qwen_vl(image_path, prompt)

        # 保存结果
        all_results.append({
            "page_num": page_num,
            "img_index": img_idx,
            "image_path": image_path,
            "content": result
        })

        # 打印结果
        print(f"\n识别结果:\n{result}\n")

    # ========== 步骤3: 保存所有识别结果到文件 ==========
    output_result_path = "pdf_all_images_recognition_result.txt"
    with open(output_result_path, "w", encoding="utf-8") as f:
        f.write(f"PDF文件: {pdf_path}\n")
        f.write(f"共提取并识别 {len(all_results)} 张嵌入图片\n\n")
        for result in all_results:
            f.write(f"========== 第 {result['page_num']} 页 / 图片 {result['img_index']} ==========\n")
            f.write(f"图片文件: {result['image_path']}\n")
            f.write(f"识别内容:\n{result['content']}\n\n")

    print("=" * 80)
    print(f"所有识别结果已保存到: {output_result_path}")
    print(f"提取的图片保存在当前目录")
    print("=" * 80)


if __name__ == "__main__":
    main()
