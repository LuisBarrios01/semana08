import pandas as pd
import numpy as np

data = pd.read_excel('BI_Alumnos08.xlsx')
from sklearn import linear_model

#Prueba de indicadores
from sklearn.metrics import mean_squared_error, r2_score

lista = pd.DataFrame()

altura = data['Altura']
edad = data['Edad']

lista['Altura'] = altura
lista['Edad'] = edad

variables = np.array(lista)

xpeso = data['Peso'].values

#Metodo de regresión lineal multiple
RLM = linear_model.LinearRegression()
RLM.fit(variables, xpeso)

#Saber si hay relacion entre variables + 0.5
I_Pred = RLM.predict(variables)
print()
print('Coeficiencde de R: ', RLM.coef_)
print('Termino Independiente: ', RLM.intercept_)
print('Error de cuadradado medio: %.2f' %  mean_squared_error(xpeso,I_Pred))
print('Puntaje de varianza: %.2f ' % r2_score(xpeso, I_Pred))

Pred_Peso = RLM.predict([[180, 29]])
print('Prediccion de peso: ', int(Pred_Peso))