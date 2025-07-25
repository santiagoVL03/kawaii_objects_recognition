import os
import tensorflow as tf
import numpy as np
from PIL import Image


def create_robust_dataset(images_dir, masks_dir, target_shape_img, target_shape_mask, 
                         batch_size=16, validation_split=0.2, seed=123):
    """
    Crea datasets de entrenamiento y validación de forma robusta, 
    manejando diferentes formatos de imagen y validando los datos
    """
    
    # Obtener y validar rutas de archivos
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Encontradas {len(image_files)} imágenes y {len(mask_files)} máscaras")
    
    # Verificar que hay el mismo número de imágenes y máscaras
    assert len(image_files) == len(mask_files), f"Número diferente de imágenes ({len(image_files)}) y máscaras ({len(mask_files)})"
    
    # Crear rutas completas
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]
    
    # Validar que los archivos existen
    valid_pairs = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))
        else:
            print(f"Archivo no encontrado: {img_path} o {mask_path}")

    print(f"{len(valid_pairs)} pares válidos de imagen-máscara")
    
    # Separate the valid paths
    valid_image_paths = [pair[0] for pair in valid_pairs]
    valid_mask_paths = [pair[1] for pair in valid_pairs]
    
    # Calculate train and validation sizes
    dataset_size = len(valid_pairs)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    # Create indices and shuffle them
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Divide into train and validation
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_image_paths = [valid_image_paths[i] for i in train_indices]
    train_mask_paths = [valid_mask_paths[i] for i in train_indices]
    val_image_paths = [valid_image_paths[i] for i in val_indices]
    val_mask_paths = [valid_mask_paths[i] for i in val_indices]

    print(f"Entrenamiento: {len(train_image_paths)} muestras")
    print(f"Validación: {len(val_image_paths)} muestras")

    def preprocess_robust(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, dtype=tf.uint8, expand_animations=False)
        
        image = tf.ensure_shape(image, [None, None, 3])
        image = tf.image.resize(image, [target_shape_img[0], target_shape_img[1]])
        image = tf.cast(image, tf.float32) / 255.0
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, dtype=tf.uint8, expand_animations=False)
        
        mask = tf.ensure_shape(mask, [None, None, 1])
        mask = tf.image.resize(mask, [target_shape_mask[0], target_shape_mask[1]], 
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.int32)
        
        # mask = mask - 1 # Our actual dataset has masks starting from 0 we donot need this
        
        mask = tf.clip_by_value(mask, 0, 2)
        
        mask = tf.squeeze(mask, axis=-1)
        
        return image, mask
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
    
    train_dataset = train_dataset.map(preprocess_robust, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_robust, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def test_dataset_creation(images_dir, masks_dir, target_shape_img, target_shape_mask):
    """
    Función para probar la creación del dataset y verificar que funcione correctamente
    """
    print("Probando creación de dataset...")
    
    try:
        train_ds, val_ds = create_robust_dataset(
            images_dir, masks_dir, target_shape_img, target_shape_mask,
            batch_size=2, validation_split=0.2
        )
        
        print("Dataset creado exitosamente!")
        
        for batch_idx, (images, masks) in enumerate(train_ds.take(1)):
            print(f"Forma del batch de imágenes: {images.shape}")
            print(f"Forma del batch de máscaras: {masks.shape}")
            
            unique_values = tf.unique(tf.reshape(masks, [-1]))[0].numpy()
            print(f"Valores únicos en máscaras: {unique_values}")
            print(f"Rango de valores de imagen: [{tf.reduce_min(images).numpy():.3f}, {tf.reduce_max(images).numpy():.3f}]")
            break
        
        return train_ds, val_ds
        
    except Exception as e:
        print(f"Error al crear dataset: {str(e)}")
        return None, None


def preview_robust_dataset(dataset, num_samples=2):
    """
    Función mejorada para visualizar muestras del dataset
    """
    import matplotlib.pyplot as plt
    
    try:
        sample_batch = next(iter(dataset))
        images, masks = sample_batch
        
        print(f"Forma de imágenes en el batch: {images.shape}")
        print(f"Forma de máscaras en el batch: {masks.shape}")
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, images.shape[0])):
            # Mostrar imagen
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Imagen {i+1}')
            axes[i, 0].axis('off')
            
            mask_display = masks[i].numpy()
            unique_values = np.unique(mask_display.flatten())  # Aplanar antes de unique
            axes[i, 1].imshow(mask_display, cmap='tab10', vmin=0, vmax=2)
            axes[i, 1].set_title(f'Máscara {i+1} (clases: {unique_values})')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error al crear preview: {str(e)}")
        return None


def quick_dataset_setup(project_root_dir, batch_size=16, validation_split=0.2):
    """
    Configuración rápida del dataset para entrenamiento
    """
    images_dir = os.path.join(project_root_dir, "segmentation_dataset_balanced", "train", "images")
    masks_dir = os.path.join(project_root_dir, "segmentation_dataset_balanced", "train", "masks")
    
    target_shape_img = [128, 128, 3]
    target_shape_mask = [128, 128, 1]

    print(f"Directorio de imágenes: {images_dir}")
    print(f"Directorio de máscaras: {masks_dir}")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Directorio de imágenes no encontrado: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"Directorio de máscaras no encontrado: {masks_dir}")
    
    train_ds, val_ds = create_robust_dataset(
        images_dir, masks_dir, 
        target_shape_img, target_shape_mask,
        batch_size, validation_split
    )
    
    return train_ds, val_ds
