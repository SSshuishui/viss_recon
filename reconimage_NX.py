import numpy as np
import pandas as pd
    
def main():
    # 读取原始数据
    l = pd.read_csv('lmn10M/l10M.txt', header=None, names=['l'])
    m = pd.read_csv('lmn10M/m10M.txt', header=None, names=['m'])
    n = pd.read_csv('lmn10M/n10M.txt', header=None, names=['n'])
    lmn = pd.concat([l, m, n], axis=1)
    del l, m, n

    # 读取去重数据
    l1 = pd.read_csv('lmn10M/l10M_dup.txt', header=None, names=['l'])
    m1 = pd.read_csv('lmn10M/m10M_dup.txt', header=None, names=['m'])
    n1 = pd.read_csv('lmn10M/n10M_dup.txt', header=None, names=['n'])
    values = pd.read_csv('torch_10M/combined.txt', header=None, names=['value'])
    lmn_dup = pd.concat([l1, m1, n1, values], axis=1)
    del l1, m1, n1, values


    print("lmn.shape = ", lmn.shape)    # (438483600, 3)
    print("lmn_dup.shape = ", lmn_dup.shape)  # (396005546, 3)

    # 合并
    merge_df = pd.merge(lmn, lmn_dup, on=['l','m','n'], how='left')
    # 检查
    last_column_has_nan = merge_df.iloc[:, -1].isna().any()
    if last_column_has_nan:
        print("最后一列包含NaN值。")
    else:
        print("最后一列不包含NaN值。")
    
    merge_df['value'].to_csv('lmn10M/period1.txt', header=False, index=False)
    print("保存完成！")

if __name__ == "__main__":
    main()