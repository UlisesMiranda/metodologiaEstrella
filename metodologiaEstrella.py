import numpy
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


df = pd.read_csv('nuevo_dataset.csv', header=0)

columnas = list(df.columns)
columna_target = columnas[6]

# Selecciona las columnas hasta el penúltimo atributo
columnas_comparar = df.columns[:-1]
print(columnas_comparar)


# df = df.sample(frac=1).reset_index(drop=True)

# Encuentra las filas duplicadas
duplicados = df[df.duplicated(subset=columnas_comparar, keep=False)]
print(duplicados)

# # Agrupa las filas duplicadas por sus valores
# grupos_duplicados = duplicados.groupby(list(duplicados.columns))

# # Imprime las filas duplicadas y sus duplicados correspondientes
# for grupo, indices in grupos_duplicados.groups.items():
#     filas = df.loc[indices]
#     print(f"\nGrupo de duplicados para {grupo}:\n{filas}")
    
# Filtra las filas duplicadas con valores distintos en el último atributo
duplicados_distintos_ultimo = duplicados[duplicados.duplicated(subset=columnas_comparar, keep=False) | duplicados.duplicated(subset=[df.columns[-1]], keep=False)]

# Agrupa las filas duplicadas por sus valores hasta el penúltimo atributo
grupos_duplicados = duplicados_distintos_ultimo.groupby(list(columnas_comparar))


intersecciones = []
# Imprime las filas duplicadas con valores distintos en el último atributo
for grupo, indices in grupos_duplicados.groups.items():
    filas = df.loc[indices]
    print(f"\nGrupo de duplicados con valores distintos en el último atributo para {grupo}:\n{filas}")
    
    if filas.values[0][-1] != filas.values[1][-1]:
        intersecciones.append(filas.values)
        print("interseccion")

print(intersecciones)

    
# # choose positive kernel
# kernel_pos = df.index[0]
# target_class = df[df.columns[-1]][kernel_pos]

#  # df with target class the same as positive kernel
# df_pos = df[df[df.columns[-1]] == target_class]
# # df with target class different from positive kernel
# df_neg = df[df[df.columns[-1]] != target_class]

# print(df_pos)
# print(df_neg)

def aprendizaje(df_pos: pd.DataFrame, df_neg: pd.DataFrame, indice: int):
    semilla = incializarSemilla(df_pos, indice)
    # semilla = ['head', 'strong', 'yes']
    
    # print(f"Semilla:\n({','.join(semilla[:-1])}) -> [{semilla[-1]}]")
    print("Semilla", semilla)
    
    # semillaCaracteristicas = []
    # for i, element in enumerate(semilla):
    #     semillaCaracteristicas.append([columns[i], element])
        
    # print(semillaCaracteristicas)
    
    # Obtén todas las combinaciones posibles
    combinaciones_posibles = obtener_combinaciones(semilla)
    
    # # Imprime las combinaciones guardadas
    # for combinacion in combinaciones_posibles:
    #     print(f'[{combinacion}]')

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
    
    # print(all(elemento in semilla for elemento in combinacion))
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
    # Obtén el array que cumple con las condiciones
        result = min(expresionesConCobertura, key=lambda x: (len(x[0][0]), -x[1]))
        # print("Array que cumple con las condiciones:", result)
    else:
        result = [['Hubo una interseccion'], None]
    
    return result[0]
    

def obtener_combinaciones(entrada, index=0, combinacion_actual=[]):
    combinaciones = []

    if index == len(entrada):
        if combinacion_actual:
            combinaciones.append(combinacion_actual)
        return combinaciones

    # Incluye el elemento actual en la combinación
    combinaciones.extend(obtener_combinaciones(entrada, index + 1, combinacion_actual + [entrada[index]]))

    # Excluye el elemento actual de la combinación
    combinaciones.extend(obtener_combinaciones(entrada, index + 1, combinacion_actual))

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
    print(targetClass)
    
    df_pos = df[df[df.columns[-1]] == targetClass]
    df_neg = df[df[df.columns[-1]] != targetClass]
    print(df_pos)
    print(df_neg)
    
    df_pos = df_pos.sample(frac=1)
    print(df_pos)
    
    complejoResultante = []
    for i in range(df_pos.shape[0]):
        complejoResultante.append(aprendizaje(df_pos, df_neg, i))
        
    resultadosFinales = list(set(tuple(array) for array in complejoResultante))
        
    resultado = f'{' V '.join(str(regla) for regla in resultadosFinales)} -> {targetClass}'
    
    print(resultado)
    
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
                y_pred.append("NO")

    print(len(y_pred))
    print(len(df_nuevo))
    
    print(y_pred)
    y_true = list(df.iloc[:, -1])
    print(y_true)
   
    # Matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)
    print(f'Matriz de Confusión:\n{confusion_mat}')

    # Reporte de clasificación
    classification_rep = classification_report(y_true, y_pred)
    print(f'Reporte de Clasificación:\n{classification_rep}')
    
algoritmoSTAR(df, -1)
    
