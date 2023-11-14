import csv
from itertools import product

# Definir el dataset original
dataset = [
    ["Stab", "Error", "Sign", "Wind", "Mag", "Vis", "Auto-pilot"],
    ['stab', 'SS', '*', '*', 'Low', 'yes', 'YES'],
    ['stab', 'SS', '*', '*', 'Med', 'yes', 'YES'],
    ['stab', 'SS', '*', '*', 'Strong', 'yes', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Low', 'yes', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Med', 'yes', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Low', 'yes', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Med', 'yes', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Strong', 'yes', 'YES'],
    ['*', '*', '*', '*', '*', 'no', 'YES'],
    ['stab', 'MM', 'pp', 'head', 'Strong', 'yes', 'NO'],
    ['xstab', '*', '*', '*', '*', 'yes', 'NO'],
    ['stab', 'LX', '*', '*', '*', 'yes', 'NO'],
    ['stab', 'XL', '*', '*', '*', 'yes', 'NO'],
    ['stab', 'MM', 'nn', 'tail', '*', 'yes', 'NO'],
    ['*', '*', '*', '*', 'Out', 'yes', 'NO']
]

# Dominio de valores
domain = {
    "Stab": ["stab", "xstab"],
    "Error": ["XL", "LX", "MM", "SS"],
    "Sign": ["pp", "nn"],
    "Wind": ["head", "tail"],
    "Mag": ["Low", "Medium", "Strong", "Out"],
    "Vis": ["yes", "no"],
    "Auto-pilot": ["YES", "NO"]
}

# Función para generar todas las combinaciones
def generate_combinations(row, domain):
    combinations = []
    for i, value in enumerate(row):
        if value == "*":
            key = dataset[0][i]  # Encuentra el nombre de la columna
            possible_values = domain[key]
            combinations.append(possible_values)
        else:
            combinations.append([value])
    return list(product(*combinations))

# Crear un nuevo dataset con todas las combinaciones
new_dataset = [dataset[0]]  # Encabezados
for row in dataset[1:]:
    if "*" in row:
        combinations = generate_combinations(row, domain)
        new_dataset.extend([list(combination) for combination in combinations])
    else:
        new_dataset.append(row)

# Escribir el nuevo dataset en un archivo CSV
with open("nuevo_dataset.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(new_dataset)