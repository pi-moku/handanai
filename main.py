import pandas as pd
from sklearn import tree
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')

#データ読み込み
df = pd.read_csv('test.csv')

#特徴量
xcol = ['one','two','three']
x = df[xcol]
t = df['four']

# AかBか判断するため決定木
model = tree.DecisionTreeClassifier(random_state=0)
model.fit(x,t)

nai = [[3,2,1]]
pre = model.predict(nai)
if pre == ['A']:
    print('A')
    # AなのでA予測(回帰)
    #Aのデータ読み込み
    df_a = pd.read_csv('a.csv')
    #特徴量
    xcol_a = ['one','two','three']
    x_a = df_a[xcol_a]
    t_a = df_a['four']
    ax_train,ax_test,ay_train,ay_test = train_test_split(x_a,t_a,test_size=0.2, random_state=0)
    model_a = LinearRegression()
    model_a.fit(ax_train,ay_train)
    print(model_a.predict(nai))

elif pre == ['B']:
    print('B')
    # BなのでB予測(回帰)
    #Aのデータ読み込み
    df_b = pd.read_csv('b.csv')
    #特徴量
    xcol_b = ['one','two','three']
    x_b = df_b[xcol_b]
    t_b = df_b['four']
    bx_train,bx_test,by_train,by_test = train_test_split(x_b,t_b,test_size=0.2, random_state=0)
    model_b = LinearRegression()
    model_b.fit(bx_train,by_train)
    print(model_b.predict(nai))
