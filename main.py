import os
import math
import sympy 
import pymorphy2
import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
from sklearn import preprocessing
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from scipy.stats import shapiro, kstest
from scipy.stats import uniform as sp_rand
from numpy.random import seed
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_extraction.text import HashingVectorizer
from statsmodels.stats.outliers_influence import variance_inflation_factor




# Путь к папке с входными данными
db_path='D:/PythonProjects/R_D/data' 



''' ==> ГЕНЕРАТОР ДАТАСЕТОВ

На входе : порядковый номер этапа / доля обучающей выборки / индикатор вызова функции, представляющей текстовое значение в виде n-мерного числового вектора
На выходе: тренировочный и тестовый датасеты (признаки и метки отдельно для каждого)

        Датасеты формируются по итерационному принципу: DF = [ DF_0(X), DF_1(df_0), ... , DF_k(DF(k-1)) ], где:
            DF_0 = [X, Y(0)]
            DF_1 = [DF_0, Y(1)]
             ...
            DF_k = [DF_k-1), Y(k)] 
            X - исходная матрица признаков, Y(i) - результирующий вектор на i-ом этапе, i=[0..k), k - количество наблюдений в датасете
    
       По мере прохождения этапов, к обучающему датасету прибавляются(конкатинируются) столбцы с метками классов всех предыдущих этапов,
       выступающих уже в роли признаков для классификации на текущем этапе. Такой же подход планируется и при классификации новых данных.
       
       На этапе представления текста в виде n-мерного вектора применяется метод PCA для выявления Одной главной компоненты вектора

'''
#def df_getData(task_n=0, tr_share=0.8, str2vec=False):
#   
#    # Разбиваем сформированный датасет на обучающий и тестовый
#    def getData_TrainTest(df, ti, str2vec=False):
#        
#        # Заголовки для столбцов, разделяющих выборку. Base_XcolumnsCount - количество признаков в исходных данных
#        delimiter_X, delimiter_Y = Columns_all[Base_XcolumnsCount + ti - 1], Columns_all[Base_XcolumnsCount + ti]  
#        # Разделяем датасет на признаки и классы
#        X, Y = df.loc[:, :delimiter_X], df.loc[:, delimiter_Y:]    
#
#
#        # РАЗДЕЛЯЕМ ДАТАСЕТ НА ОБУЧАЮЩИЙ И ТЕСТОВЫЙ (ОБЯЗАТЕЛЬНО ДО ТРАНСФОРМАЦИИ векторов, с целью предотвращения утечки информации на шаге применения PCA)
#        tr_count = np.min(int(len(df) * tr_share),0)     # Доля обучающей выборки
#        x_train, x_test = X[:tr_count], X[tr_count:]     # Обучающая и Тестовая выборки для признаков
#        y_train, y_test = Y[:tr_count], Y[tr_count:]     # Обучающая и Тестовая выборки для меток
#
#        
#        # Если требуется преобразовать текст в вектор, то делаем это
#        if str2vec:
#            x_train = df_txt2vec(x_train)   # метки и тестовые данные не трогаем
#            x_test  = df_txt2vec(x_test )   # метки и обучающие данные не трогаем
#
#        return x_train, y_train, x_test, y_test
#   
#    
#    # По умолчанию формируем всевозможные датасеты (по длине словаря этапов) иначе выводим датасет только для указанного этапа
#    lenTasks=len(dict_Tasks) # количество столбцов, требующих классификации (кол-во этапов классификации)
#    to_task = task_n if task_n >= 0 else lenTasks - 1     
#    
#    
#    # Цикл по всем этапам
#    DF = pd.DataFrame()
#    for ti in range(0, to_task): # ti - индекс текущей задачи. Совпадает с индексам этапов в словаре "dict_Tasks" 
#        
#        # Для первого элемента обучающего датасета берутся матрица признаков X и вектор значений Y(t0).
#        if ti == 0:
#            DF = pd.concat([X,  Y[ dict_Tasks[ti]] ], axis=1) # Cоздаем требуемый датасет (имя столбца с меткой берем по текущему индексу<=>ключу из словаря этапов)
#        else:
#            DF = pd.concat([DF, Y[ dict_Tasks[ti]] ], axis=1) # Последующие датасеты формируются из матрицы полученной на прошлом шаге: DF(t(i-1)) и вектора значений = Y(ti)
#        
#        
#        if to_task == lenTasks: # Если указанное число совпадает с кол-вом столбцов из меток, то:
#            yield getData_TrainTest(DF, ti, str2vec)  # последовательно возвращаем датасеты для каждом этапе, вплоть до Задачи с указанным порядковым номером "ti"
#        
#        elif ti == to_task: # Если указанное число совпадает с текущим, то:
#            yield getData_TrainTest(DF, ti, str2vec)  # возвращаем датасет для текущего этапа(метки)
#        else:
#            pass

# X, Y = df_txt2vec(X,Y) # ТЕКСТ В ЧИСЛО

## ГЕНЕРАТОР ДАТАСЕТОВ 
## После разделения на тренировочный и тестовый датасеты к текстовым признакам применяется алгоритм HashingVectorizer их представления в числовом виде.
#DF_generator = df_getData(task_n=0, tr_share=0.7, str2vec=True)  # "task_n" - номер этапа классификации ( нумерация с 0, при -1 берутся все исходные данные)
#
## Последовательно формируем датасет для каждого этапа (для каждого столбца с метками)
#iterr = 0 # Текущая иттерация
#for dfg in DF_generator:     # Последовательно получаем датасеты для каждого из этапов
#        # Извлекаем для текущего этапа компоненты датасета
#        x_train, y_train, x_test, y_test = dfg[0].copy(),dfg[1].copy(),dfg[2].copy(),dfg[3].copy()
#        iterr+=1
#        break # Вернет датасет для первого этапа (для первого столбца с метками)

  
''' ==>   РАЗБИВАЕМ НА ОБУЧАЮЩИЕ И ТЕСТОВЫЕ ДАННЫЕ 
'''

# Разбиваем сформированный датасет на обучающий и тестовый
def getX_TrainTest(df, tr_share=0.8, str2vec=False):
    
    # Заголовок для столбцов, разделяющих выборку. Base_XcolumnsCount - количество признаков в исходных данных
    delimiter_X = Columns_all[Base_XcolumnsCount - 1]
    
    # Выделяем признаки
    X = df.loc[:, :delimiter_X]   

   # Разделяем входные данные на тренировочные и тестовые (ОБЯЗАТЕЛЬНО ДО ТРАНСФОРМАЦИИ векторов, с целью предотвращения утечки информации на шаге применения PCA)
    tr_count = np.min(int(len(df) * tr_share),0)     # Доля обучающей выборки
    x_train, x_test = X[:tr_count], X[tr_count:]     # Обучающая и Тестовая выборки для признаков

    #  Восстонавливаем индексы после разделения
    x_test.index = pd.RangeIndex(len(x_test.index))
    
    # Если требуется преобразовать текст в вектор, то делаем это
    if str2vec:
        x_train = df_txt2vec(x_train)   # тестовые данные не трогаем
        x_test  = df_txt2vec(x_test )   # обучающие данные не трогаем

    return x_train, x_test
   
    


# Разбиваем сформированный датасет на обучающий и тестовый
def getY_TrainTest(df, tr_share=0.8, str2vec=False):
    
     # Заголовок для столбцов, разделяющих выборку. Base_XcolumnsCount - количество признаков в исходных данных
    delimiter_Y =  Columns_all[Base_XcolumnsCount] 
    
    # Выделяем метки
    Y = df.loc[:, delimiter_Y:]    

    # Разделяем входные данные на тренировочные и тестовые (ОБЯЗАТЕЛЬНО ДО ТРАНСФОРМАЦИИ векторов, с целью предотвращения утечки информации на шаге применения PCA)
    tr_count = np.min(int(len(df) * tr_share),0)     # Доля обучающей выборки
    y_train, y_test = Y[:tr_count], Y[tr_count:]     # Обучающая и Тестовая выборки для меток

    #  Восстонавливаем индексы после разделения
    y_test.index = pd.RangeIndex(len(y_test.index))

    # Если требуется преобразовать текст в вектор, то делаем это
    if str2vec:
        y_train = df_txt2vec(y_train)   # тестовые данные не трогаем
        y_test  = df_txt2vec(y_test )   # обучающие данные не трогаем

    return y_train, y_test
   
    

            
''' ==>   ПРЕОБРАЗОВЫВАЕМ ТЕКСТ 
'''
# Преобразуем столбцы с текстом в числовой вектор. 
# Для упрощения задачи, в качестве первого приближения была выбранна одна единственная компонента (n_components=1).
            
def df_txt2vec(df):
    start_time = time()
    df = df.copy()

    lenX = len(list(df)) # Количество строк в датасете
    
    print('\n>> [df_txt2vec]: Start')
    for i, xtr in enumerate(df): 
        vec = txt2vec( df[xtr], n_features = Base_nfuture)      # Преобразовываем столбец с текстом в список многомерных векторов
        vec = lowerDimensions_PCA(vec, n_components=1)         # Снижаем размерность методом главных компонент
        df[ list(df)[i] ] = pd.DataFrame(vec)
        print('\r>> [df_txt2vec]: Прогресс: %1f%% ' % ((i+1)/(lenX)*100), end='')
        
#    # Добавляем в список полученный вектор текста меток (название столбца получаем по индексу текущей итерации)
#    vec = txt2vec( Y[list(Y)[0]].map(str), n_features = Base_nfuture)
#    vec = lowerDimensions_PCA(vec, n_components=1)
#    Y[ list(Y)[0] ] = pd.DataFrame(vec)   
    
    TIME_TO_FIT = time() - start_time
    print("\n>> [df in DF_generator]: Получен за %s сек. ---" % (TIME_TO_FIT))

    return df 


# Преобразовываем столбец текстовых данных в n-мерный числовой вектор [На входе: Series]
def txt2vec(df, n_features=0):
    n_features=n_features if n_features > 0 else Base_nfuture  # Размерность вектора. По умолчанию, размер устанавливается равный числу уникальных символов во всем наборе данных
    coder = HashingVectorizer(tokenizer=f_tokenizer, n_features= n_features)
    vec = coder.fit_transform(df.tolist()).toarray()
    return vec


# Отрисовываем одномерный массив
def plot_1D(ar,lbl="1D",yl1=-1,yl2=1):
    plt.plot(ar, "o",label=lbl)
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.ylim(yl1, yl2)
    plt.show()
    
    
# Функция очистки url от префиксов (в целях снижения шума)
def clear_url(s):
    replace_values = ['http://','https://','www.']
    for v in replace_values:
        s = s.replace(v,'')
    return s


# Функция токенезирования
def f_tokenizer(s):
    morph = pymorphy2.MorphAnalyzer()
    t = s.split(' ') if type(s) == 'unicode' else s
    f = []
    for j in t:
        m = morph.parse(j.replace('.',''))
        if len(m) != 0:
            wrd = m[0]
            f.append(wrd.normal_form)
    return f


# Подсчет числа уникальных токенов (символов)  [На входе: DataFrame/Series/List/String]
def getTokens_uniq(df):
    tokens=[]
    for elm in df:
        row_txt_uniq = list(set([y for y in elm])) # отбор уникальных токенов Значения признака
        tokens.extend(row_txt_uniq)
    return list(set(tokens)) # возвращаем список уникальных токенов среди всех значений признака



# СНИЖЕНИЕ РАЗМЕРНОСТИ. PCA
def lowerDimensions_PCA(df, n_components=3):
    pca = PCA(n_components = n_components) # n_components = 0.95 - 95% дисперсии
    pca.fit(df)
    return pca.transform(df)



# ПРОВЕРКА ПРИЗНАКОВ НА МУЛЬТИКОЛЛЕНИАРНОСТЬ
def df_notMulticoll(X):
    X=X.copy().fillna(0)
    # Линейная комбинация 
    _, inds = sympy.Matrix(X).rref()      # "inds" содердит индексы линейно независемых признаков
    not_multicol = pd.DataFrame()
    not_multicol['Features'] = X.columns                     # Названия признаков
    not_multicol['Linear independence'] = 0                  # Линейно зависимые
    not_multicol['Linear independence'].loc[list(inds)] = 1  # Линейно независимые

    # Коэффициент вздутия дисперсии (>7-10 признак мультиколлениарности)
    not_multicol["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Отбираем признаки при срабатывании любого из критериев
    good =  not_multicol.loc[(not_multicol['Linear independence'] == 1) | (not_multicol['VIF'] >= 20) ]['Features']

    return X[good], not_multicol




#   ПРЕДОБРАБОТКА ДАННЫХ
def dataPreparation(df,df_len=0):
    callback = df.copy()
 
    # Заменяем Nan на прочерки
    NaNs=[] 
    for f in callback:
        features = callback[f]
        logs=[f, False, 0] # Список содержит: Название столбца, Результат проверки на NaN, Кол-во NaN после обнаружения нулей и применения метода fillna
        if features.isnull().values.any() or features.isna().values.any():
            callback[f] = callback[f].fillna('—')
            NaN_count = features.isna().sum().sum() # Подсчитываем количество пустых элементов, но работает медленнее
            logs=[f, True, NaN_count]
        NaNs.append(logs) # Записываем результат проверки
    
    
    #  Очищаем данные от мусора
    replace_columns_url = ['Источник','Страница звонка','Страница входа']
    for u in replace_columns_url:
        callback[u] = callback[u].apply(clear_url)
    
    #  Очистка дублирующей информации в столбцах ( удаляем в полях "Страница звонка" и "Страница входа" содержимое поля "Домен вызова". Если ничего не осталось, то ставим прочерк)
    #  Применять аккуратно, т.к. это влияет на значимость признака при его отборе
    callback['Страница звонка'] = callback.apply(lambda x: x['Страница звонка'].replace( x['Домен вызова']+'/','')  
                                                        if x['Страница звонка'][0:-1] != x['Домен вызова'] else '—', axis=1) 
    callback['Страница входа'] = callback.apply(lambda x: x['Страница входа'].replace( x['Домен вызова']+'/','')  
                                                        if x['Страница входа'][0:-1] != x['Домен вызова'] else '—', axis=1) 
    #  Разделяем сцепленные слова
    callback['UTM'] = callback['UTM'].str.replace('_',' ').str.replace('-',' ').str.replace('|',' ')
    callback['Страница звонка'] = callback['Страница звонка'].str.replace('_',' ').str.replace('-',' ').str.replace('/',' ')
    callback['Страница входа'] = callback['Страница входа'].str.replace('_',' ').str.replace('-',' ').str.replace('/',' ') 
    
    #  Приводим весь текст к нижнему регистру
    for c in callback:
        callback[c] = callback[c].astype(str).str.lower()
        
       
    #  Удаляем дубликаты строк
    callback = callback.drop_duplicates()

    #  Ограничиваем набор данных
    if df_len>0:
        callback = callback.loc[0:df_len-1,:] 
        
    #  Восстонавливаем индексы после удаления дубликатов
    callback.index = pd.RangeIndex(len(callback.index))
    
    return callback













''' ==>   ЗАГРУЗКА ДАННЫХ
'''

# Загрузка исходных данных
callback = pd.read_csv(os.path.join(db_path,'callback.csv' ), sep=',')
 



''' ==>   ПРЕДОБРАБОТКА ДАННЫХ
'''
callback = dataPreparation(callback, df_len=5000) # df_len=0 - весь набор данных




''' ==> РАЗДЕЛЯЕМ ИСХОДНЫЕ ДАННЫЕ на признаки и классы
'''
#  Количество признаков в исходных данных
Base_XcolumnsCount = 6

#  Список заголовков в исходных данных (требуется для формирования датасетов в функции "df_getData" )
Columns_all = list(callback)

#  Заголовки для столбцов, разделяющих датасет на признаки и метки. Base_XcolumnsCount - количество признаков в исходных данных
X_lastcol, Y_firstcol = Columns_all[Base_XcolumnsCount - 1], Columns_all[Base_XcolumnsCount]  

#  РАЗДЕЛЯЕМ ИСХОДНЫЕ ДАННЫЕ на признаки и классы 
X, Y = callback.loc[:, :X_lastcol ], callback.loc[:, Y_firstcol:]  

#  Список заголовков в разделенных данных (требуется для формирования датасетов в функции "df_getData" )
Columns_X, Columns_Y = list(X), list(Y)





''' ==>  ВСПОМОГАТЕЛЬНЫЕ СПИСКИ, СЛОВАРИ И КОНСТАНТЫ
'''

#  БАЗОВОЕ КОЛ-ВО КОМПОНЕНТ вектора полученного из текста
Base_nfuture = int(len(getTokens_uniq(callback))) # кол-во уникальных символов в исходных данных


#  СПИСОК УНИКАЛЬНЫХ МЕТОК (в разрезе по задачам)
labels=[]
for col in Y:
    labels.append(list( Y[col].unique() ))
del col 

#  СЛОВАРЬ С ЭТАПАМИ КЛАССИФИКАЦИИ (формируется из заголовков матрицы Y)
dict_Tasks = {} 
for i,k in enumerate(list(Y)):
    dict_Tasks[i] = k
del i,k 





''' ==> ФОРМИРУЕМ ДАТАСЕТЫ
'''

#  МАТРИЦА ПРИЗНАКОВ (ТЕКСТ=>ЧИСЛО)
x_train, x_test = getX_TrainTest(callback, tr_share=0.8, str2vec = True)

#  МАТРИЦА МЕТОК (ТЕКСТ=>ТЕКСТ)
y_train, y_test = getY_TrainTest(callback, tr_share=0.8, str2vec = False)




#  СОХРАНЯЕМ ДАТАСЕТ (последний из сформированных)
pd.DataFrame(x_train).to_csv(os.path.join(db_path,'x_train.csv'))
pd.DataFrame(x_test).to_csv(os.path.join(db_path,'x_test.csv'))
pd.DataFrame(y_train).to_csv(os.path.join(db_path,'y_train.csv'))
pd.DataFrame(y_test).to_csv(os.path.join(db_path,'y_test.csv'))



# Загрузка датасета
x_train = pd.read_csv(os.path.join(db_path,'x_train.csv' ), sep=',')
y_train = pd.read_csv(os.path.join(db_path,'y_train.csv' ), sep=',')
x_test  = pd.read_csv(os.path.join(db_path,'x_test.csv' ), sep=',')
y_test = pd.read_csv(os.path.join(db_path,'y_test.csv' ), sep=',')
x_train = x_train.iloc[:,1:] 
y_train = y_train.iloc[:,1:] 
x_test  = x_test.iloc[:,1:] 
y_test = y_test.iloc[:,1:] 



# ПРОВЕРКА И УДАЛЕНИЕ МУЛЬТИКОЛЛЕНИАРНЫХ ПРИЗНАКОВ (через коэф-т вздутия дисперсии и проверки на линейную комбинацию)
x_train, notMulticoll_tr = df_notMulticoll(x_train)
x_test,  notMulticoll_te = df_notMulticoll(x_test )



#''' ==> ПРОВЕРКА НА НОРМАЛЬНОСТЬ РАСПРЕДЕЛЕНИЯ
#'''
#norm=[]
#seed(1) # фиксируем генератор
#alpha = 0.05 # уровень недоверия
#for c in list(winequality)[:-2]:
#    data = winequality[c]
#    #plt.hist(data) # гистограмма распределения
#    
#    # Теста Шапиро-У (до 2000 элементов)
#    stat1, p1 = shapiro(data)
#    T1='Sample looks Gaussian (fail to reject H0)' if p1 > alpha else 'Sample does not look Gaussian (reject H0)'
#    norm.append([stat1,p1,T1])
#    
#    #    # Критерий Колмогорова-Смирнова (от 2000 элементов)
#    #    stat2, p2 = kstest(data, 'norm')
#    #    T2='Sample looks Gaussian (fail to reject H0)' if p2 > alpha else 'Sample does not look Gaussian (reject H0)'
#    #    norm.append([[stat1,p1,T1],[stat2,p2,T2]])

## СТАНДАРТИЗАЦИЯ (надо перенести в pipeline)
#x_train_std = preprocessing.scale(x_train)
#x_test_std  = preprocessing.scale(x_test)







# Инициируем различные модели
def initModel(m, params):
    if   m == 0:
        model = LogisticRegression(**params[m]) 
    elif m == 1:
        model = KNeighborsClassifier(n_neighbors=5)
    elif m == 2:
        model = DecisionTreeClassifier(**params[m]) 
    elif m == 3:
        model = RandomForestClassifier(**params[m])
    elif m == 4:
        model = SVC(**params[m])
    elif m == 5:
        model = SVC(**params[m])
    elif m == 6:
        model = AdaBoostClassifier(**params[m])
    elif m == 7:
        model = GradientBoostingClassifier(**params[m])
    elif m == 8:
        model = MLPClassifier(**params[m]) 
    else:
        model = GaussianNB(**params[m])
    return model


algoritm_names=['LogistReg','kNN','Tree','Forest','SVM_line','SVM_rbf','AdaBoost','GradBoost','Network','NBaes']

''' ==> ЗАДАЕМ ГИПЕРПАРАМЕТРЫ АЛГОРИТМОВ
'''

alpha_ls = dict(alpha = 0.05)
n_neighbors_ls = dict(n_neighbors = 5)
solver_ls = dict(solver='lbfgs')
n_estimators_ls = dict(n_estimators = 100)
max_depth_ls = dict(max_depth = 2)
random_state_ls = dict(random_state = 1618)
C1_ls = dict(C = 1.0)
C10_ls = dict(C = 1.0)
gamma_ls = dict(gamma = 0.1)
hidden_layer_sizes_ls = dict(hidden_layer_sizes = (5, 2))
kernel_rbf_ls = dict(kernel= 'rbf')
kernel_linear_ls = dict(kernel = 'linear')
leaf_size_ls =dict(leaf_size=30)
penalties_ls = dict(penalties = 'l1') #, 'l2', 'elasticnet', 'none']

p0 = reduce(lambda x,y: dict(x, **y), (solver_ls, random_state_ls))
p1 = reduce(lambda x,y: dict(x, **y), (n_neighbors_ls, leaf_size_ls))
p2 = dict()
p3 = reduce(lambda x,y: dict(x, **y), (n_estimators_ls, max_depth_ls, random_state_ls))
p4 = reduce(lambda x,y: dict(x, **y), (kernel_linear_ls , C1_ls, random_state_ls))
p5 = reduce(lambda x,y: dict(x, **y), (kernel_rbf_ls, C10_ls, random_state_ls ,gamma_ls ))
p6 = reduce(lambda x,y: dict(x, **y), (n_estimators_ls , random_state_ls))
p7 = reduce(lambda x,y: dict(x, **y), (n_estimators_ls , random_state_ls))
p8 = reduce(lambda x,y: dict(x, **y), (solver_ls, random_state_ls, alpha_ls, hidden_layer_sizes_ls))
p9 = dict()
params=[p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]



''' ==>  АПРОБАЦИЯ АЛГОРИТМОВ КЛАССИФИКАЦИИ
'''


# Списки с результатами тестов моделей
T_confusion_matrixs = []
T_accuracy_scores = []


for l in range(0, len(Columns_Y)):


    # Трансформируем Pandas в Numpy. 
    x_tr, y_tr = np.asarray(x_train), np.asarray(list(y_train.iloc[:,l].map(str)))
    x_te, y_te = np.asarray(x_test ), np.asarray(list(y_test. iloc[:,l].map(str)))


    # Списки с результатами тестов моделей
    confusion_matrixs = []
    accuracy_scores = []
    

    # Проверка всех заданных моделей
    for m in range(0,10):
        
        # Инициируем требуемую модель
        model = initModel(m,params)
        print(l, Columns_Y[l],' -> ', m, algoritm_names[m])
        
        #  Обучаем
        model.fit(x_tr, y_tr)
        
        # Применяем модель к тестовым данным
        expected = y_te
        predicted = model.predict(x_te)
        
#        confusion_matrix=metrics.confusion_matrix(expected, predicted)
        accuracy=metrics.accuracy_score(expected, predicted)
        
        # Записываем результаты теста алгоритма
#        confusion_matrixs.append(confusion_matrix)
        accuracy_scores.append(accuracy) 
#        print(metrics.classification_report(expected, predicted))
#        print(confusion_matrix)


    # Записываем результаты теста для метки
#    T_confusion_matrixs.append(confusion_matrixs)
    T_accuracy_scores.append(accuracy_scores)




T_accuracy_scores_df = pd.DataFrame(T_accuracy_scores, columns=algoritm_names, index=Columns_Y)




''' НЕОБХОДИМО ДОДЕЛАТЬ: 
    
 >>  1. Создать конвейер Pipeline, передав ему список необходимых этапов.
        Например: для модели SVC, вероятно потребуются масштабирование данных, а для класса RandomForestClassifier предварительная обработка не требуется.
 >>  2. Задать сетку параметров для подбора гиперпараметров моделей.
 >>  3. Реализовать перекрестную проверку
 >>  4. Визуализировать результаты 
'''   
  

## Заготовка для будущего конвейера
#
#
#pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC()) ])
#    
#param_grid = [{
#             'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
#             'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#             'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},{
#             'classifier': [RandomForestClassifier(n_estimators=100)],
#             'preprocessing': [None], 'classifier__max_features': [1, 2, 3] }]    
#   
#for name, estimator in zip(models,clfs):
#    clf = GridSearchCV(estimator, params[name], scoring='accuracy', refit='True', n_jobs=-1, cv=5)
#    clf.fit(X_train, y_train)
#    print("best params: " + str(clf.best_params_))
#    print("best scores: " + str(clf.best_score_))
#    y_pred = clf.predict(X_test)
#    acc = accuracy_score(y_test, y_pred)
#    print("Accuracy: {:.4%}".format(acc))
#    print(classification_report(y_test, y_pred, digits=4))






seeds=1618              # Регулируем значения псевдогенератора случайных чисел
confusion_matrixs = []




# ОТБОР ПРИЗНАКОВ. Метод 1
model = ExtraTreesClassifier(random_state=seeds)
model.fit(x_tr, y_tr)
print(model.feature_importances_)


# ОТБОР ПРИЗНАКОВ. Метод 2
model = LogisticRegression(random_state=seeds)
# create the RFE model and select 3 attributes
rfe = RFE(model, 2)
rfe = rfe.fit(x_tr, y_tr)
print(rfe.support_)
print(rfe.ranking_)

# Оба метода отбора признаков говорят о минимальном вкладе 2 и 4 компоненты.
# Для текущего этапа(классификация первого столбца в справочнике). Можно их исключить, но у меня не так много признаков. Пока оставим.





seeds=1618  # Регулируем значения псевдогенератора случайных чисел


''' ==>  ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
'''
# Часто используется для задач бинарной классификации, но допускается и многоклассовая классификация методом "one-vs-all". 
# Достоинством этого алгоритма являеся то, что на выходе для каждого обьекта мы имеем вероятсность принадлежности классу.

model = LogisticRegression(random_state=1618, solver='lbfgs')
model.fit(x_tr, y_tr)
#print(model)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)

z=metrics.accuracy_score(expected, predicted)


''' ==>  НАИВНЫЙ БАЙЕС
'''
# Основной задачей алгоритма является восстановление плотностей распределения данных обучающей выборки. 
# В наивном подходе делается предположение о независимости случайных величин. Зачастую, этот метод дает хорошее качество в задачах многоклассовой классификации. 

model = GaussianNB()
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)



''' ==>  МЕТОД kNN (K-ближайших соседей)
'''
# Метод kNN часто используется как составная часть более сложного алгоритма классификации. 
# Например, его оценку можно использовать как признак для обьекта. А иногда, простой kNN на хорошо подобранных признаках дает отличное качество. 

model = KNeighborsClassifier(n_neighbors=5)
print(model)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)


''' ==>  ДЕРЕВЬЯ РЕШЕНИЙ
'''
# Classification and Regression Trees (CART) часто используются в задачах, в которых обьекты имеют категориальные признаки и используется для задач регресии и классификации.
# Очень хорошо деревья подходят для многоклассовой классификации.

model = DecisionTreeClassifier()
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)



''' ==>  СЛУЧАЙНЫЙ ЛЕС
'''
#Случайные леса - это метод обучения ансамбля, который подгоняет несколько деревьев решений к подмножествам данных и усредняет результаты. 

model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)



''' ==>  МЕТОД ОПОРНЫХ ВЕКТОРОВ
'''
# SVM (Support Vector Machines) применяется в основном для задачи классификации. Суть метода заключается в максимизации расстояния от разделяющей поверхности до ближайших экземпляров класса.
# Также как и логистическая регрессия, SVM допускает многоклассовую классификацию методом one-vs-all.

model = SVC(kernel='linear', C=1.0, random_state=1)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)




''' ==>  AdaBoostClassifier
'''
# AdaBoost  работает путем взвешивания экземпляров в наборе данных тем, насколько легко или сложно классифицировать их, позволяя алгоритму 
# оплачивать или меньше уделять им внимание при построении последующих моделей.

num_trees = 30
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)



''' ==>  Stochastic Gradient Boosting Classification (Gradient Boosting Machine)
'''
# Обладает хорошей производительностью.ми.

seed = 7
num_trees = 2
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)




''' ==>  НЕЛИНЕЙНАЯ КЛАССИФИКАЦИЯ
'''
# Нелинейность достигается за счет выбора ядра

model = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
model.fit(x_tr, y_tr)
expected = y_te
predicted = model.predict(x_te)
confusion_matrix=metrics.confusion_matrix(expected, predicted)
confusion_matrixs.append(confusion_matrix)
print(metrics.classification_report(expected, predicted))
print(confusion_matrix)




''' ==>  НЕЙРОННЫЕ СЕТИ
'''
# По сути - модный перебор. В данном примере добавил подбор гиперпараметров двумя методами

alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1618)

# Поиск по сетке
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(x_tr, y_tr)
print('>> Best__score GridSearch:',grid.best_score_)
print('>> Param_value GridSearch:',grid.best_estimator_.alpha)

# Поиск путем генерации случайных величин
param_grid = {'alpha': sp_rand()} 
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(x_tr, y_tr)
print('>> Best__score RandomizedSearch:',rsearch.best_score_)
print('>> Param_value RandomizedSearch:',rsearch.best_estimator_.alpha)





 



model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1618)
model.fit(x_tr, y_tr)

# Применяем модель к тестовым данным
expected = y_te
predicted = model.predict(x_te)

#        confusion_matrix=metrics.confusion_matrix(expected, predicted)
accuracy=metrics.accuracy_score(expected, predicted)
print(accuracy)




# МЫСЛИ в СЛУХ:

## ОПРЕДЕЛИТЕЛЬ НЕ КВАД.МАТРИЦЫ (приближаю минорами с шириной равной числу признаков)
#  На графике видно, что на значительной части последовательно идущих интервалов определитель равен нулю => данные на этом участке являются линейно зависемыми. 
#  Поскольку эта линейная зависемость не обнаружилась на этапе проверки мультиколлениарных признаков, то дело не в столбцах, а в строках (хотя на этом участке столбец тоже линейно зависем). 
#  Здесь можно, либо на каждой череде идущих подряд нулевых детерминантов производить отбор признаков(обнулять менее значимые в целях экономии ресурсов), либо вернуться на предыдущий этап и произвести очистку от мультиколлениарных строк(применить тот же метод, но к транспонированной матрице)
#  Наврено, правильнее всего будет произвести глобальную очистку от линейно зависемых строк (столбцов не транспонированной матрицы) и затем ещё раз пройтись минорами (т.к. высота минора ограничена количеством признаков, то после удаления строк в ширину окна могут попсать новые комбинации, детерминант которых тоже будет равен нулю).

#h=len(list(x_train))
#det=[]
#rank = []
#ln=math.ceil(len(x_train)/h)
#for i in range(0, ln):
#    d =  np.linalg.det( x_train.loc[i:i+h-1,] ) 
#    rk = np.linalg.matrix_rank(x_train.loc[i:i+h-1,])
#    det.append(d)
#    rank.append(rk)
#det_np=np.asarray(det)

#plot_1D(det_np,lbl="1D",yl1=det_np.min(),yl2=det_np.max()) 
#plot_1D(np.asarray(rank),lbl="1D",yl1=np.asarray(rank).min(),yl2=np.asarray(rank).max())  
















