"""

Configuración de datasets para evaluaciones
Permite cambiar entre datasets modificando una sola variable

"""


# CONFIGURACIÓN GLOBAL


# Cambiar esta variable para alternar entre datasets
DATASET_ACTIVO = "mendeley_enfermedades"  # Opciones: "enfermedades" | "mendeley_enfermedades"

# ============================================================
# DEFINICIÓN DE DATASETS
# ============================================================

DATASETS = {
    "enfermedades": {
        "id": "kaggle_enfermedades",
        "nombre": "KAGGLE - Diagnóstico de Enfermedades",
        "descripcion": "Dataset de 133 enfermedades con 20 síntomas binarios",
        "path": "../datasets/kaggle_enfermedades/gold/kaggle_dataset.csv",
        "target": "prognosis",
        "n_clases": 133,
        "n_features": 20,
        "n_muestras": 2564,
        "fuente": "https://www.kaggle.com/datasets/shobhit043/diseases-and-their-symptoms",
    },
    
    "mendeley_enfermedades": {
        "id": "mendeley_enfermedades",
        "nombre": "MENDELEY - Diagnóstico de Enfermedades",
        "descripcion": "Dataset de 25 enfermedades con 20 síntomas tras preprocesamiento",
        "path": "../datasets/mendeley_enfermedades/gold/mendeley_dataset.csv",
        "target": "prognosis",
        "n_clases": 24,
        "n_features": 20,
        "n_muestras": 358,
        "fuente": "https://data.mendeley.com/datasets/rjgjh8hgrt/6"
    }
}

# FUNCIONES DE UTILIDAD

def get_dataset_config():
    """
    Devuelve la configuración del dataset activo
    
    Returns:
        dict: Configuración completa del dataset
    """
    if DATASET_ACTIVO not in DATASETS:
        raise ValueError(f"Dataset '{DATASET_ACTIVO}' no existe. Opciones: {list(DATASETS.keys())}")
    
    return DATASETS[DATASET_ACTIVO]

def get_all_datasets():
    """
    Devuelve configuración de todos los datasets disponibles
    
    Returns:
        dict: Diccionario con todos los datasets
    """
    return DATASETS

def set_active_dataset(dataset_id):
    """
    Cambia el dataset activo
    
    Args:
        dataset_id (str): ID del dataset a activar
    """
    global DATASET_ACTIVO
    if dataset_id not in DATASETS:
        raise ValueError(f"Dataset '{dataset_id}' no existe. Opciones: {list(DATASETS.keys())}")
    DATASET_ACTIVO = dataset_id

def print_dataset_info():
    """
    Imprime información del dataset activo
    """
    config = get_dataset_config()
    
    print(f"DATASET ACTIVO: {config['nombre']}")
    print(f"ID:           {config['id']}")
    print(f"Descripción:  {config['descripcion']}")
    print(f"Path:         {config['path']}")
    print(f"Target:       {config['target']}")
    print(f"\nDIMENSIONES:")
    print(f"  Clases:     {config['n_clases']}")
    print(f"  Features:   {config['n_features']}")
    print(f"  Muestras:   {config['n_muestras']}")

# ============================================================
# EJECUCIÓN DIRECTA (para testing)
# ============================================================

if __name__ == "__main__":
    print("\n🔹 DATASET 1:")
    print_dataset_info()
    
    print("\n\n🔹 DATASET 2:")
    set_active_dataset("disease_symptom")
    print_dataset_info()