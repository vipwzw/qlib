#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻情感因子分析项目 - 快速启动脚本
用于快速测试项目功能
"""

import os
import sys
from pathlib import Path

# 添加utils路径以使用配置加载器
sys.path.append(str(Path(__file__).parent / "utils"))
try:
    from config_loader import ConfigLoader, create_env_file_template
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           新闻情感量化因子分析项目                           ║
    ║           BTC/USDT News Sentiment Factor Analysis            ║
    ║                                                              ║
    ║           基于qlib框架的加密货币新闻情感因子分析             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查项目依赖"""
    print("🔍 检查项目依赖...")
    
    required_packages = [
        'pandas', 'numpy', 'yaml', 'requests', 'feedparser', 'qlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有基础依赖检查通过！")
    return True

def setup_directories():
    """创建必要的目录"""
    print("\n📁 创建项目目录...")
    
    directories = [
        "data/raw/price",
        "data/raw/news",
        "data/processed",
        "data/factors",
        "data/results",
        "logs"
    ]
    
    project_root = Path(__file__).parent
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   📂 {directory}")
    
    print("✅ 目录创建完成！")

def setup_env_config():
    """设置环境变量配置"""
    print("\n🔧 检查环境变量配置...")
    
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example_file = project_root / ".env.example"
    
    # 创建.env.example模板（如果不存在）
    if HAS_CONFIG_LOADER and not env_example_file.exists():
        try:
            create_env_file_template(str(env_example_file))
        except Exception as e:
            print(f"   ⚠️ 创建.env.example失败: {e}")
    
    # 检查是否存在.env文件
    if not env_file.exists():
        print("   ❌ .env文件不存在")
        if env_example_file.exists():
            print(f"   📝 请复制 .env.example 为 .env 并配置API密钥")
            print("   命令: cp .env.example .env")
        else:
            print("   📝 请创建 .env 文件并配置API密钥")
        return False
    
    # 验证必需的环境变量
    if HAS_CONFIG_LOADER:
        try:
            loader = ConfigLoader()
            required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
            validation_results = loader.validate_required_env_vars(required_vars)
            
            all_set = True
            for var, is_set in validation_results.items():
                status = "✅" if is_set else "❌"
                print(f"   {status} {var}: {'已配置' if is_set else '未配置'}")
                if not is_set:
                    all_set = False
            
            if all_set:
                print("✅ 环境变量配置完整！")
                return True
            else:
                print("   ⚠️ 请在 .env 文件中配置缺失的API密钥")
                return False
                
        except Exception as e:
            print(f"   ⚠️ 环境变量验证失败: {e}")
            return False
    else:
        print("   ✅ .env文件存在（请确保已正确配置）")
        return True

def run_demo():
    """运行演示"""
    print("\n🚀 运行项目演示...")
    
    try:
        # 导入项目模块
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        from scripts.data_collection import DataCollectionManager
        from scripts.factor_construction import FactorBuilder
        from scripts.run_backtest import NewsSentimentBacktest
        
        print("   ✅ 模块导入成功")
        
        # 测试配置加载
        try:
            manager = DataCollectionManager()
            print("   ✅ 配置文件加载成功")
        except Exception as e:
            print(f"   ⚠️ 配置文件加载失败: {e}")
        
        # 测试因子构建器
        try:
            builder = FactorBuilder()
            print("   ✅ 因子构建器初始化成功")
        except Exception as e:
            print(f"   ⚠️ 因子构建器初始化失败: {e}")
        
        # 测试回测器
        try:
            backtester = NewsSentimentBacktest()
            print("   ✅ 回测器初始化成功")
            
            # 运行简单回测演示
            print("\n📊 运行回测演示...")
            performance = backtester.run_simple_backtest()
            
            if performance:
                print("   ✅ 回测演示完成！")
            
        except Exception as e:
            print(f"   ⚠️ 回测器测试失败: {e}")
        
    except Exception as e:
        print(f"   ❌ 演示运行失败: {e}")
        return False
    
    return True

def show_next_steps():
    """显示下一步操作指南"""
    print("\n📋 下一步操作指南:")
    print("="*50)
    print("0. 安装依赖包:")
    print("   pip install -r requirements.txt")
    print("   (这将自动安装本地qlib开发版本和其他依赖)")
    print("")
    print("1. 配置API密钥 (使用.env文件):")
    print("   a) 复制环境变量模板: cp .env.example .env")
    print("   b) 编辑 .env 文件，填入真实的API密钥:")
    print("      - BINANCE_API_KEY=your_api_key")
    print("      - BINANCE_SECRET_KEY=your_secret_key")
    print("      - 其他可选配置...")
    print("")
    print("2. 数据采集:")
    print("   python scripts/data_collection.py --data-type all --days 30")
    print("")
    print("3. 构建因子:")
    print("   python scripts/factor_construction.py")
    print("")
    print("4. 运行回测:")
    print("   python scripts/run_backtest.py --mode simple")
    print("")
    print("5. 查看详细文档:")
    print("   - 项目使用指南.md - 完整使用指南（推荐）")
    print("   - README.md - 项目说明")
    print("   - 新闻量化因子分析需求文档.md - 详细需求")
    print("   - 技术实现方案示例.py - 核心实现")
    print("")
    print("🔗 更多帮助:")
    print("   - Qlib文档: https://qlib.readthedocs.io/")
    print("   - 项目GitHub: (如果有的话)")
    print("")
    print("💡 提示:")
    print("   本项目使用本地qlib开发版本，无需单独安装qlib")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 创建目录
    setup_directories()
    
    # 设置环境变量配置
    env_configured = setup_env_config()
    
    # 运行演示
    demo_success = run_demo()
    
    # 显示下一步指南
    show_next_steps()
    
    if demo_success and env_configured:
        print("\n🎉 项目快速启动完成！所有组件运行正常。")
        print("现在您可以根据上面的指南继续配置和使用项目。")
    elif demo_success and not env_configured:
        print("\n✅ 项目基础功能正常，但需要配置API密钥。")
        print("请按照上面的指南配置 .env 文件中的API密钥。")
    else:
        print("\n⚠️ 项目启动完成，但某些组件可能需要额外配置。")
        print("请查看上面的错误信息并按照指南进行配置。")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 