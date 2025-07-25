#!/usr/bin/env python3
"""
Script de prueba para verificar la carga del modelo Keras
"""

import os
import tensorflow as tf
import numpy as np

# Forzar CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

print("=== PRUEBA DE CARGA DEL MODELO ===")
print(f"TensorFlow versión: {tf.__version__}")
print(f"Dispositivos disponibles: {tf.config.list_physical_devices()}")

def test_model_loading():
    """Prueba la carga del modelo"""
    model_path = 'best_model.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return False
    
    try:
        print(f"Intentando cargar modelo: {model_path}")
        
        # Forzar CPU explícitamente
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(model_path, compile=False)
        
        print("Modelo cargado exitosamente")
        print(f"Entrada esperada: {model.input_shape}")
        print(f"Salida esperada: {model.output_shape}")
        
        # Prueba con datos dummy
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        print("Probando predicción con datos dummy...")
        
        with tf.device('/CPU:0'):
            prediction = model.predict(dummy_input, verbose=0)
        
        print(f"Predicción exitosa. Shape: {prediction.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 El modelo se puede cargar y usar correctamente!")
    else:
        print("\n💥 Hay problemas con la carga del modelo.")
