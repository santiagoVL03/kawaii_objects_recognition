"""
Script de prueba para verificar que el dataset funcione correctamente
Ejecuta este archivo para probar la nueva implementación antes del entrenamiento completo
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))

from unet.utils.dataset_robust import test_dataset_creation, preview_robust_dataset, quick_dataset_setup
import matplotlib.pyplot as plt

def main():
    print("PROBANDO EL NUEVO SISTEMA DE DATASET EFICIENTE")
    print("=" * 50)
    
    # Configuración
    target_shape_img = [128, 128, 3]
    target_shape_mask = [128, 128, 1]
    
    # Rutas
    project_root = os.path.abspath(os.path.join(".."))
    images_dir = os.path.join(project_root, "segmentation_dataset_balanced", "train", "images")
    masks_dir = os.path.join(project_root, "segmentation_dataset_balanced", "train", "masks")
    
    print(f"Buscando imágenes en: {images_dir}")
    print(f"Buscando máscaras en: {masks_dir}")
    
    # Verificar que los directorios existan
    if not os.path.exists(images_dir):
        print(f"ERROR: No se encontró el directorio de imágenes: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"ERROR: No se encontró el directorio de máscaras: {masks_dir}")
        return
    
    print("\n PROBANDO CREACIÓN DE DATASET...")
    
    # Probar la creación del dataset
    train_ds, val_ds = test_dataset_creation(
        images_dir, masks_dir, 
        target_shape_img, target_shape_mask
    )
    
    if train_ds is None or val_ds is None:
        print("FALLO: No se pudo crear el dataset")
        return

    print("\n CREANDO PREVIEW DEL DATASET...")

    # Crear preview
    fig = preview_robust_dataset(train_ds, num_samples=2)
    
    if fig is not None:
        # Guardar preview
        preview_dir = os.path.join(project_root, "segmentation_dataset_balanced", "proccess")
        os.makedirs(preview_dir, exist_ok=True)
        
        fig.savefig(os.path.join(preview_dir, "dataset_test_preview.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Preview guardado en: {os.path.join(preview_dir, 'dataset_test_preview.png')}")
    
    print("\n PROBANDO CONFIGURACIÓN RÁPIDA...")
    
    try:
        # Probar configuración rápida
        train_quick, val_quick = quick_dataset_setup(project_root, batch_size=8, validation_split=0.2)
        print("Configuración rápida exitosa!")
        
        # Contar batches
        train_batches = sum(1 for _ in train_quick)
        val_batches = sum(1 for _ in val_quick)
        print(f"Batches de entrenamiento: {train_batches}")
        print(f"Batches de validación: {val_batches}")
        
    except Exception as e:
        print(f"Error en configuración rápida: {e}")
        return
    
    print("\n" + "=" * 50)
    print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    print("El sistema está listo para entrenar sin problemas de memoria")
    print("\n Para entrenar el modelo, use:")
    print("   python3 final_train.py")

if __name__ == "__main__":
    main()
