#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能运行脚本
自动检测和准备数据，然后运行新闻情感因子策略
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入项目模块
from scripts.auto_data_manager import AutoDataManager
from scripts.run_backtest import NewsSentimentBacktest


class SmartRunner:
    """智能运行器"""
    
    def __init__(self):
        """初始化智能运行器"""
        self.project_root = project_root
        print("🚀 新闻情感因子分析 - 智能运行器")
        print("="*50)
    
    def run(self, config_path: str = "configs/config.yaml", 
            mode: str = "simple",
            force_download: bool = False,
            skip_data_check: bool = False,
            save_results: bool = True):
        """运行完整流程"""
        
        start_time = time.time()
        
        try:
            print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 项目路径: {self.project_root}")
            print(f"⚙️ 配置文件: {config_path}")
            print(f"🎯 运行模式: {mode}")
            print("-" * 50)
            
            # 步骤1：数据管理
            if not skip_data_check:
                print("\n📋 第1步：数据完整性检查")
                self._manage_data(config_path, force_download)
            else:
                print("\n⏭️ 跳过数据检查步骤")
            
            # 步骤2：运行策略回测
            print("\n🎯 第2步：运行策略回测")
            results = self._run_strategy(config_path, mode, save_results)
            
            # 步骤3：总结
            print("\n📊 第3步：运行总结")
            self._print_summary(results, start_time)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _manage_data(self, config_path: str, force_download: bool = False):
        """管理数据"""
        try:
            data_manager = AutoDataManager(config_path)
            
            if force_download:
                print("🔄 强制重新下载所有数据...")
                
                # 强制下载价格数据
                success = data_manager.download_price_data()
                if not success:
                    print("⚠️ 价格数据下载失败，将尝试使用现有数据")
                
                # 强制下载新闻数据
                success = data_manager.download_news_data()  
                if not success:
                    print("⚠️ 新闻数据下载失败，将尝试使用现有数据")
                
                # 重新生成因子
                success = data_manager.generate_factors()
                if not success:
                    print("⚠️ 因子生成失败")
                    
            else:
                # 智能检查和下载
                success = data_manager.ensure_data_ready()
                if not success:
                    print("⚠️ 数据准备过程中出现问题，将尝试继续运行")
            
            # 显示数据摘要
            print("\n📊 数据摘要:")
            summary = data_manager.get_data_summary()
            for data_type, info in summary.items():
                if 'error' not in info and info:
                    print(f"  {data_type}: {info.get('record_count', 'N/A')} 条记录")
                    
        except Exception as e:
            print(f"⚠️ 数据管理过程出错: {e}")
            print("将尝试使用现有数据继续运行...")
    
    def _run_strategy(self, config_path: str, mode: str, save_results: bool = True):
        """运行策略"""
        try:
            print(f"🎯 启动{mode}模式回测...")
            
            # 不进行自动数据检查，因为已经在前面完成了
            backtester = NewsSentimentBacktest(config_path, auto_data_check=False)
            
            if mode == "workflow":
                results = backtester.run_workflow_backtest()
            else:
                results = backtester.run_simple_backtest()
            
            if save_results and results is not None:
                print("💾 保存回测结果...")
                if isinstance(results, dict):
                    # 如果结果是性能指标字典，转换为DataFrame
                    import pandas as pd
                    results_df = pd.DataFrame([results])
                    backtester.save_results(results_df)
                else:
                    backtester.save_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ 策略运行失败: {e}")
            return None
    
    def _print_summary(self, results, start_time: float):
        """打印运行总结"""
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 运行完成!")
        print(f"⏱️ 总耗时: {duration:.2f} 秒")
        print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results:
            if isinstance(results, dict):
                print(f"📈 策略收益: {results.get('total_return', 0):.2%}")
                print(f"📊 夏普比率: {results.get('sharpe_ratio', 0):.3f}")
            print("✅ 回测结果已生成")
        
        print("\n💡 使用提示:")
        print("  - 查看详细结果: data/results/ 目录")
        print("  - 查看数据文件: data/ 目录下各子文件夹")
        print("  - 修改配置: configs/config.yaml")
        print("  - 强制重新下载数据: python smart_run.py --force-download")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻情感因子分析智能运行器")
    
    # 基本参数
    parser.add_argument("--config", default="configs/config.yaml", 
                       help="配置文件路径")
    parser.add_argument("--mode", choices=["simple", "workflow"], default="simple",
                       help="运行模式: simple=简单回测, workflow=qlib工作流")
    
    # 数据管理参数
    parser.add_argument("--force-download", action="store_true",
                       help="强制重新下载所有数据")
    parser.add_argument("--skip-data-check", action="store_true",
                       help="跳过数据检查和下载")
    
    # 输出参数
    parser.add_argument("--no-save", action="store_true",
                       help="不保存回测结果")
    
    # 工具功能
    parser.add_argument("--check-data", action="store_true",
                       help="只检查数据状态，不运行策略")
    parser.add_argument("--data-summary", action="store_true",
                       help="显示数据摘要信息")
    parser.add_argument("--clean-data", type=int, metavar="DAYS",
                       help="清理N天前的旧数据文件")
    
    args = parser.parse_args()
    
    try:
        runner = SmartRunner()
        
        # 工具功能
        if args.check_data:
            data_manager = AutoDataManager(args.config)
            availability = data_manager.check_data_availability()
            
            print("\n📋 数据可用性检查:")
            print("-" * 30)
            print(f"价格数据: {'✅ 可用' if availability['price_data'] else '❌ 缺失'}")
            print(f"新闻数据: {'✅ 可用' if availability['news_data'] else '❌ 缺失'}")
            print(f"因子数据: {'✅ 可用' if availability['factor_data'] else '❌ 缺失'}")
            
            return 0
        
        elif args.data_summary:
            data_manager = AutoDataManager(args.config)
            summary = data_manager.get_data_summary()
            
            print("\n📊 数据摘要报告:")
            print("=" * 50)
            
            for data_type, info in summary.items():
                print(f"\n{data_type.upper().replace('_', ' ')}:")
                if 'error' in info:
                    print(f"  ❌ 错误: {info['error']}")
                else:
                    for key, value in info.items():
                        print(f"  {key}: {value}")
            
            return 0
        
        elif args.clean_data:
            data_manager = AutoDataManager(args.config)
            success = data_manager.clean_old_data(args.clean_data)
            return 0 if success else 1
        
        else:
            # 运行完整流程
            save_results = not args.no_save
            success = runner.run(
                config_path=args.config,
                mode=args.mode,
                force_download=args.force_download,
                skip_data_check=args.skip_data_check,
                save_results=save_results
            )
            
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断运行")
        return 1
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 