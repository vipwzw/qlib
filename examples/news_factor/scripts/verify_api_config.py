#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API 配置验证脚本
用于验证 API Key 和网络连接是否正常
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def main():
    print("🔍 验证 DeepSeek API 配置...")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查 API Key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY 未设置")
        print("\n📝 请按以下步骤配置:")
        print("1. 在项目根目录创建 .env 文件")
        print("2. 在文件中添加: DEEPSEEK_API_KEY=your-api-key-here")
        print("3. 重新运行此脚本")
        return 1
    
    print(f"✅ API Key 已设置: {api_key[:10]}...")
    
    # 检查其他配置
    base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    print(f"✅ Base URL: {base_url}")
    print(f"✅ Model: {model}")
    
    # 测试 API 连接
    print("\n🧪 测试 API 连接...")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=10
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, please respond with 'API test successful'"}],
            max_tokens=20,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API 连接成功!")
        print(f"   响应: {result}")
        print(f"   消耗 tokens: {response.usage.total_tokens}")
        
        return 0
        
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        print("\n🔧 可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 验证 API Key 是否正确")
        print("3. 确认 DeepSeek 服务是否可用")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 