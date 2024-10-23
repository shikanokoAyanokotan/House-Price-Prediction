import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def DataVisualization(df):
    # SalePrice statistics
    print(df['SalePrice'].describe())

    # Distribution graph of SalePrice
    sns.displot(df['SalePrice'])
    plt.show()

    # Correlation matrix 
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corrmat = df[numeric_columns].corr()
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.show()

    # Correlation matrix of the top 10 features with the highest correlation coefficients with 'SalePrice'
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm10 = np.corrcoef(df[cols].values.T)
    sns.set_theme(font_scale=1.25)
    hm = sns.heatmap(cm10, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, 
                    xticklabels=cols.values)
    plt.show()

    # GrLivArea (Ground living area) vs SalePrice
    plt.figure(figsize=(8, 6))
    plt.scatter(x=df['GrLivArea'], y=df['SalePrice'])
    plt.xlabel('Ground Living Area')
    plt.ylabel('Sale Price')
    plt.show()

    # OverallQual (overall quality) vs SalePrice
    plt.figure(figsize=(8, 6))
    plt.boxplot([df['SalePrice'][df['OverallQual'] == i] for i in sorted(df['OverallQual'].unique())])
    plt.xlabel('Overall Quality')
    plt.ylabel('Sale Price')
    plt.xticks(ticks=range(1, 11), labels=range(1, 11))
    plt.show()
