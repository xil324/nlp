from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1",
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


# ============ 翻译智能体 ============
class Translation(BaseModel):
    """文本翻译：自动识别需要翻译的文本，包括原始语种、目标语种和待翻译内容"""
    source_language: str = Field(description="原始语种（如：英语、中文、日语等）")
    target_language: str = Field(description="目标语种（如：中文、英语、法语等）")
    text_to_translate: str = Field(description="需要翻译的文本内容")
    detected_intent: Literal["translate"] = Field(description="检测到的意图，固定为翻译")

print("=" * 60)
print("翻译智能体测试")
print("=" * 60)

# 英译中
result1 = ExtractionAgent(model_name="deepseek-chat").call(
    '帮我将"good！"翻译为中文',
    Translation
)
print("测试1 - '帮我将\"good！\"翻译为中文'")
print(f"原始语种: {result1.source_language}")
print(f"目标语种: {result1.target_language}")
print(f"待翻译文本: {result1.text_to_translate}")
print(f"检测意图: {result1.detected_intent}")
print("-" * 40)

# 中译英
result2 = ExtractionAgent(model_name="deepseek-chat").call(
    '请把"你好，世界"翻译成英语',
    Translation
)
print("测试2 - '请把\"你好，世界\"翻译成英语'")
print(f"原始语种: {result2.source_language}")
print(f"目标语种: {result2.target_language}")
print(f"待翻译文本: {result2.text_to_translate}")
print(f"检测意图: {result2.detected_intent}")
print("-" * 40)

# 日译中
result3 = ExtractionAgent(model_name="deepseek-chat").call(
    '可以帮我翻译一下"こんにちは"到中文吗？',
    Translation
)
print("测试3 - '可以帮我翻译一下\"こんにちは\"到中文吗？'")
print(f"原始语种: {result3.source_language}")
print(f"目标语种: {result3.target_language}")
print(f"待翻译文本: {result3.text_to_translate}")
print(f"检测意图: {result3.detected_intent}")
print("-" * 40)

# 法译中
result4 = ExtractionAgent(model_name="deepseek-chat").call(
    '翻译"Bonjour"为中文',
    Translation
)
print("测试4 - '翻译\"Bonjour\"为中文'")
print(f"原始语种: {result4.source_language}")
print(f"目标语种: {result4.target_language}")
print(f"待翻译文本: {result4.text_to_translate}")
print(f"检测意图: {result4.detected_intent}")
print("-" * 40)

# 复杂句子翻译
result5 = ExtractionAgent(model_name="deepseek-chat").call(
    '请帮我把这句话译成英文："今天天气真好"',
    Translation
)
print("测试5 - '请帮我把这句话译成英文：\"今天天气真好\"'")
print(f"原始语种: {result5.source_language}")
print(f"目标语种: {result5.target_language}")
print(f"待翻译文本: {result5.text_to_translate}")
print(f"检测意图: {result5.detected_intent}")
print("=" * 60)

