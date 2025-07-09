#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子分析脚本
分析生成的技术因子的统计特性、相关性和有效性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self, factor_file: str = None):
        self.project_root = Path(__file__).parent.parent
        self.factor_file = factor_file or self._get_latest_factor_file()
        self.factors_df = self._load_factors()
        
    def _get_latest_factor_file(self) -> str:
        """获取最新的因子文件"""
        factor_files = glob.glob(str(self.project_root / "data" / "factors" / "*.parquet"))
        if not factor_files:
            raise FileNotFoundError("未找到因子文件")
        return sorted(factor_files)[-1]
    
    def _load_factors(self) -> pd.DataFrame:
        """加载因子数据"""
        print(f"加载因子文件: {self.factor_file}")
        df = pd.read_parquet(self.factor_file)
        print(f"因子数据形状: {df.shape}")
        return df
    
    def basic_statistics(self):
        """基础统计分析"""
        print("\n" + "="*50)
        print("因子基础统计信息")
        print("="*50)
        
        # 基本统计
        stats = self.factors_df.describe()
        print("\n基本统计指标:")
        print(stats)
        
        # 缺失值统计
        print("\n缺失值统计:")
        missing = self.factors_df.isnull().sum()
        missing_pct = (missing / len(self.factors_df)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        
        # 无穷值统计
        print("\n无穷值统计:")
        inf_counts = {}
        for col in self.factors_df.columns:
            inf_count = np.isinf(self.factors_df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            for col, count in inf_counts.items():
                print(f"{col}: {count}")
        else:
            print("无无穷值")
    
    def correlation_analysis(self):
        """相关性分析"""
        print("\n" + "="*50)
        print("因子相关性分析")
        print("="*50)
        
        # 计算相关性矩阵
        corr_matrix = self.factors_df.corr()
        
        # 找出高相关性的因子对
        print("\n高相关性因子对 (|相关系数| > 0.8):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'Factor1': corr_matrix.columns[i],
                        'Factor2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df)
        else:
            print("未发现高相关性因子对")
        
        # 绘制相关性热力图
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.1)
        plt.title('因子相关性热力图')
        plt.tight_layout()
        
        # 保存图片
        output_dir = self.project_root / "data" / "analysis"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "factor_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"\n相关性热力图已保存至: {output_dir / 'factor_correlation_heatmap.png'}")
        plt.close()
    
    def factor_distribution_analysis(self):
        """因子分布分析"""
        print("\n" + "="*50)
        print("因子分布分析")
        print("="*50)
        
        # 选择一些关键因子进行分布分析
        key_factors = ['returns_1m', 'rsi_14', 'macd', 'bb_position', 'adx', 'atr']
        available_factors = [f for f in key_factors if f in self.factors_df.columns]
        
        if not available_factors:
            print("未找到关键因子进行分析")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, factor in enumerate(available_factors):
            if i >= len(axes):
                break
            
            data = self.factors_df[factor].dropna()
            
            # 绘制分布图
            axes[i].hist(data, bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'{factor} 分布')
            axes[i].set_xlabel(factor)
            axes[i].set_ylabel('密度')
            
            # 添加统计信息
            mean_val = data.mean()
            std_val = data.std()
            skew_val = data.skew()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.4f}')
            axes[i].text(0.05, 0.95, f'标准差: {std_val:.4f}\n偏度: {skew_val:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[i].legend()
        
        # 隐藏多余的子图
        for i in range(len(available_factors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = self.project_root / "data" / "analysis"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "factor_distributions.png", dpi=300, bbox_inches='tight')
        print(f"\n因子分布图已保存至: {output_dir / 'factor_distributions.png'}")
        plt.close()
    
    def factor_time_series_analysis(self):
        """因子时间序列分析"""
        print("\n" + "="*50)
        print("因子时间序列分析")
        print("="*50)
        
        # 选择一些关键因子
        key_factors = ['returns_1m', 'rsi_14', 'bb_position', 'adx']
        available_factors = [f for f in key_factors if f in self.factors_df.columns]
        
        if not available_factors:
            print("未找到关键因子进行分析")
            return
        
        # 只取最近1000个数据点进行可视化
        recent_data = self.factors_df.tail(1000)
        
        fig, axes = plt.subplots(len(available_factors), 1, figsize=(15, 3*len(available_factors)))
        if len(available_factors) == 1:
            axes = [axes]
        
        for i, factor in enumerate(available_factors):
            data = recent_data[factor].dropna()
            axes[i].plot(data.index, data.values, linewidth=0.8)
            axes[i].set_title(f'{factor} 时间序列 (最近1000个数据点)')
            axes[i].set_ylabel(factor)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = self.project_root / "data" / "analysis"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "factor_timeseries.png", dpi=300, bbox_inches='tight')
        print(f"\n因子时间序列图已保存至: {output_dir / 'factor_timeseries.png'}")
        plt.close()
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*50)
        print("生成因子分析报告")
        print("="*50)
        
        report_content = f"""
# 因子分析报告

## 基本信息
- 因子文件: {Path(self.factor_file).name}
- 因子数量: {self.factors_df.shape[1]}
- 时间点数量: {self.factors_df.shape[0]}
- 时间范围: {self.factors_df.index.min()} 至 {self.factors_df.index.max()}

## 因子列表
{chr(10).join([f"- {col}" for col in self.factors_df.columns])}

## 数据质量
- 完整因子数量: {(self.factors_df.isnull().sum() == 0).sum()}
- 有缺失值的因子数量: {(self.factors_df.isnull().sum() > 0).sum()}
- 平均缺失率: {self.factors_df.isnull().mean().mean():.2%}

## 建议
1. 对于高相关性的因子，可以考虑只保留其中一个，减少冗余
2. 对于缺失值较多的因子，需要进一步检查数据质量
3. 建议进行因子有效性测试，评估预测能力
4. 可以考虑对因子进行标准化处理
"""
        
        # 保存报告
        output_dir = self.project_root / "data" / "analysis"
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "factor_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已保存至: {output_dir / 'factor_analysis_report.md'}")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("开始因子全面分析...")
        
        self.basic_statistics()
        self.correlation_analysis()
        self.factor_distribution_analysis()
        self.factor_time_series_analysis()
        self.generate_report()
        
        print("\n" + "="*50)
        print("因子分析完成！")
        print("="*50)
        print("生成的文件:")
        output_dir = self.project_root / "data" / "analysis"
        for file in output_dir.glob("*"):
            print(f"- {file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子分析工具")
    parser.add_argument("--factor-file", help="指定因子文件路径")
    parser.add_argument("--basic", action="store_true", help="仅进行基础统计分析")
    parser.add_argument("--correlation", action="store_true", help="仅进行相关性分析")
    parser.add_argument("--distribution", action="store_true", help="仅进行分布分析")
    parser.add_argument("--timeseries", action="store_true", help="仅进行时间序列分析")
    
    args = parser.parse_args()
    
    try:
        analyzer = FactorAnalyzer(args.factor_file)
        
        if args.basic:
            analyzer.basic_statistics()
        elif args.correlation:
            analyzer.correlation_analysis()
        elif args.distribution:
            analyzer.factor_distribution_analysis()
        elif args.timeseries:
            analyzer.factor_time_series_analysis()
        else:
            analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"❌ 因子分析失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 