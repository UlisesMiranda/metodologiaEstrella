import numpy
import pandas as pd
from itertools import combinations
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def mezclar(df: pd.DataFrame):
    num_filas = len(df)
    indices_filas = list(range(num_filas))

    # Baraja aleatoriamente los índices de filas
    random.shuffle(indices_filas)

    # Crea un nuevo DataFrame barajado
    return df.iloc[indices_filas].reset_index(drop=True)

def aprendizaje(df_pos: pd.DataFrame, df_neg: pd.DataFrame, indice: int):
    semilla = incializarSemilla(df_pos, indice)
   
    # Obtén todas las combinaciones posibles
    combinaciones_posibles = obtenerCombinaciones(semilla)

    # Imprime las combinaciones posibles
    # for combinacion in combinaciones_posibles:
    #     print(combinacion)
        
    # print(len(combinaciones_posibles))
    
    ejemplosNegativos = []
    for ejemplo in df_neg.iloc[:, :].values:
        ejemploAuxiliar = []
        for i, val in enumerate(ejemplo):
            val = f'{df.columns[i]}={val}'
            ejemploAuxiliar.append(val)
        ejemplosNegativos.append(ejemploAuxiliar)
            
    # for ejemplo in ejemplosNegativos:
    #     print(ejemplo)
    
    ejemplosPositivos = []
    for ejemplo in df_pos.iloc[:, :].values:
        ejemploAuxiliar = []
        for i, val in enumerate(ejemplo):
            val = f'{df.columns[i]}={val}'
            ejemploAuxiliar.append(val)
        ejemplosPositivos.append(ejemploAuxiliar)
            
    # for ejemplo in ejemplosPositivos:
    #     print(ejemplo)
    
    expresionesConsistentes = []
    for combinacion in combinaciones_posibles:
        for i, ejemplo in enumerate(ejemplosNegativos):
            
            if all(elemento in ejemplo for elemento in combinacion):
                # print("Incosistente->",combinacion)
                break
                
            if i == len(ejemplosNegativos) - 1:
                # print("Cosistente->",combinacion)
                expresionesConsistentes.append(combinacion)
                
    
    # print("\nExpresiones consistentes:")
    # for expresion in expresionesConsistentes:
    #     print(expresion)
        
    coberturas = []
    for combinacion in expresionesConsistentes:
        coberturaSum = 0
        for i, ejemplo in enumerate(ejemplosPositivos):
            
            if all(elemento in ejemplo for elemento in combinacion):
                coberturaSum += 1
                
        coberturas.append(coberturaSum)
    
    expresionesConCobertura = []
    # print("\nCoberturas")
    for i, combinacion in enumerate(expresionesConsistentes):
        # print(f'{combinacion} -> {coberturas[i]}')
        expresionesConCobertura.append([combinacion, coberturas[i]])
        
    # for e in expresionesConCobertura:
    #     print(e)
        
    if expresionesConCobertura:
        result = min(expresionesConCobertura, key=lambda x: (len(x[0][0]), -x[1]))
        # print("Array que cumple con las condiciones:", result)
    else:
        result = [['Hubo una interseccion'], None]
    
    return result[0]
    

def obtenerCombinaciones(entrada, index=0, combinacion_actual=[]):
    combinaciones = []

    if index == len(entrada):
        if combinacion_actual:
            combinaciones.append(combinacion_actual)
        return combinaciones

    # Incluye el elemento actual en la combinación
    combinaciones.extend(obtenerCombinaciones(entrada, index + 1, combinacion_actual + [entrada[index]]))

    # Excluye el elemento actual de la combinación
    combinaciones.extend(obtenerCombinaciones(entrada, index + 1, combinacion_actual))

    return combinaciones

def incializarSemilla(df_pos: pd.DataFrame, indice: int) -> list:
    semilla = []
    for i, val in enumerate(df_pos.iloc[indice, :-1].values):
        val_semilla = f'{df.columns[i]}={val}'
        semilla.append(val_semilla)
        
    return semilla
    

def algoritmoSTAR(df: pd.DataFrame, targetColumn):
    columns = df.columns
    df = df.sort_values(by=columns[targetColumn], ascending=False)
    print(df)
    
    # Dividimos df en positivos y negativos
    targetClass = df[df.columns[-1]][0]
    print("Clase a aprender: ", targetClass)
    
    df_pos = df[df[df.columns[-1]] == targetClass]
    df_neg = df[df[df.columns[-1]] != targetClass]
    
    elseClass = df_neg.iloc[0, -1]
    
    print("\nEjemplos positivos: \n", df_pos)
    print("\nEjemplos negativos: \n", df_neg)
    
    df_pos = mezclar(df_pos)
    print("Ejemplos positivos mezclados: \n", df_pos)
    
    complejoResultante = []
    for i in range(df_pos.shape[0]):
        complejoResultante.append(aprendizaje(df_pos, df_neg, i))
        
    resultadosFinales = list(set(tuple(array) for array in complejoResultante))
        
    resultado = f'{' V '.join(str(regla) for regla in resultadosFinales)} -> {targetClass}'
    
    print("\nHipotesis aprendida: ", resultado)
    
    df_nuevo = []
    
    for ejemplo in df.iloc[:, :].values:
        ejemploAuxiliar = []
        for i, val in enumerate(ejemplo):
            val = f'{df.columns[i]}={val}'
            ejemploAuxiliar.append(val)
        df_nuevo.append(ejemploAuxiliar)
    
    y_pred = []
    
    for fila in df_nuevo:
        for i, combinacion in enumerate(resultadosFinales):
            if all(elemento in fila for elemento in combinacion):
                y_pred.append(targetClass)
                break
            if i == len(resultadosFinales) - 1:
                y_pred.append(elseClass)

    y_true = list(df.iloc[:, -1])
   
    matrizConfusion = confusion_matrix(y_true, y_pred)
    print(f'\nMatriz de Confusión:\n{matrizConfusion}')

    reporteClasificacion = classification_report(y_true, y_pred)
    print(f'Reporte de Clasificación:\n{reporteClasificacion}')
    
    
df = pd.read_csv('nuevo_dataset.csv', header=0)
print("Dataset original: \n", df)

columnas = list(df.columns)
columna_target = columnas[6]
columnas_comparar = df.columns[:-1]

duplicados = df[df.duplicated(subset=columnas_comparar, keep=False)]

# Agrupa las filas duplicadas por sus valores hasta el penúltimo atributo
grupos_duplicados = duplicados.groupby(list(columnas_comparar))

intersecciones = []
for grupo, indices in grupos_duplicados.groups.items():
    filas = df.loc[indices]
    # print(f"\nGrupo de duplicados con valores distintos en el último atributo para {grupo}:\n{filas}")
    
    if filas.values[0][-1] != filas.values[1][-1]:
        intersecciones.append(filas.values)
        # print("interseccion")

print()
for interseccion in intersecciones:
    print("Interseccion en: \n", interseccion)

df = df.drop_duplicates(subset=columnas_comparar, keep=False)
print("\nEliminando duplicados, queda: \n", df)
    
algoritmoSTAR(df, -1)
    
