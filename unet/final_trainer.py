"""
Entrenamiento de segmentación usando el dataset eficiente y robusto
Este archivo resuelve el problema de memoria RAM usando tf.data.Dataset
"""

import sys
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping # type: ignore
sys.path.append(os.path.abspath(os.path.join("..")))

from unet.utils import model
from unet.utils.dataset_robust import quick_dataset_setup, preview_robust_dataset
import matplotlib.pyplot as plt

def main():
    print("ENTRENAMIENTO DE SEGMENTACIÓN - VERSIÓN EFICIENTE EN MEMORIA")
    print("=" * 60)
    
    batch_size = 16  # Adjust according to your RAM/GPU available
    epochs = 8
    learning_rate = 1e-4
    validation_split = 0.2
    
    # Configure TensorFlow for GPU (if available)
    setup_tensorflow()
    
    project_root = os.path.abspath(os.path.join(".."))
    output_dir = os.path.join(project_root, "segmentation_dataset_balanced", "proccess")
    model_dir = os.path.join("model")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Directorio de salida: {output_dir}")
    print(f"Directorio de modelos: {model_dir}")
    
    print("\nCREANDO DATASETS...")
    try:
        train_dataset, val_dataset = quick_dataset_setup(
            project_root, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
        print("Datasets creados exitosamente!")
        
    except Exception as e:
        print(f"ERROR al crear datasets: {e}")
        return
    
    # Crear preview del dataset
    print("\nCREANDO PREVIEW DEL DATASET...")
    try:
        fig = preview_robust_dataset(train_dataset, num_samples=3)
        if fig:
            fig.savefig(os.path.join(output_dir, "training_dataset_preview.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Preview guardado en: {output_dir}/training_dataset_preview.png")
    except Exception as e:
        print(f"⚠️ No se pudo crear preview: {e}")
    
    # Crear y compilar modelo
    print("\nCREANDO MODELO UNET...")
    try:
        unet = model.UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3)
        print("Modelo creado!")
        
        unet.summary()
        
        unet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # type: ignore
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # type: ignore
            metrics=['accuracy']
        )
        print("Modelo compilado!")
        
    except Exception as e:
        print(f"ERROR al crear modelo: {e}")
        return
    
    print("\nCONFIGURANDO CALLBACKS...")
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    print("Callbacks configurados!")
    
    print(f"\n🎯 INICIANDO ENTRENAMIENTO ({epochs} épocas)...")
    print("💡 Tip: El entrenamiento usará muy poca RAM gracias a tf.data.Dataset")
    
    try:
        # Entrenar usando datasets eficientes
        history = unet.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nENTRENAMIENTO COMPLETADO!")
        
    except Exception as e:
        print(f"ERROR durante entrenamiento: {e}")
        return
    
    # Guardar modelo final
    print("\nGUARDANDO MODELO FINAL...")
    try:
        unet.save(os.path.join(model_dir, "final_unet_model.keras"))
        print(f"Modelo final guardado en: {model_dir}/final_unet_model.keras")
    except Exception as e:
        print(f"Error guardando modelo final: {e}")

    # Crear gráficos de entrenamiento
    print("\nCREANDO GRÁFICOS DE ENTRENAMIENTO...")
    try:
        create_training_plots(history, output_dir)
        print(f"Gráficos guardados en: {output_dir}")
    except Exception as e:
        print(f"Error creando gráficos: {e}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print(f"Resultados en: {output_dir}")
    print(f"Modelos en: {model_dir}")
    print("El modelo está listo para predicciones")


def setup_tensorflow():
    print("Configurando TensorFlow...")
    
    import os
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detectada: {len(gpus)} dispositivo(s)")
            print("XLA deshabilitado para evitar problemas de compatibilidad")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Usando CPU para entrenamiento")

    # Configurar nivel de log
    tf.get_logger().setLevel('ERROR')


def create_training_plots(history, output_dir):
    """Crear gráficos del historial de entrenamiento"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Historial de Entrenamiento - Segmentación UNet', fontsize=16)
    
    # Gráfico de pérdida
    axes[0, 0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida del Modelo')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico de precisión
    axes[0, 1].plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    axes[0, 1].set_title('Precisión del Modelo')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Precisión')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (si está en el historial)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Tasa de Aprendizaje')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nno disponible', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Diferencia entre train y val loss
    if len(history.history['loss']) > 0:
        diff = [abs(t - v) for t, v in zip(history.history['loss'], history.history['val_loss'])]
        axes[1, 1].plot(diff, linewidth=2, color='red')
        axes[1, 1].set_title('Diferencia |Train Loss - Val Loss|')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Diferencia Absoluta')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history_complete.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Crear un gráfico simple adicional
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
