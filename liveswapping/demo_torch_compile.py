#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Демонстрационный скрипт для тестирования torch.compile() ускорения.

Этот скрипт тестирует различные режимы torch.compile() и показывает
потенциальное ускорение для LiveSwapping моделей.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple

def check_torch_compile_support() -> bool:
    """Проверить поддержку torch.compile()."""
    try:
        if hasattr(torch, 'compile'):
            torch_version = torch.__version__
            major, minor = map(int, torch_version.split('.')[:2])
            return major >= 2
        return False
    except Exception:
        return False


def create_dummy_face_swap_model():
    """Создать простую модель для тестирования."""
    class DummyFaceSwapModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Имитация архитектуры face swap модели
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 512)
            )
            
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(512, 128 * 4 * 4),
                torch.nn.ReLU(),
                torch.nn.Unflatten(1, (128, 4, 4)),
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                torch.nn.Sigmoid()
            )
        
        def forward(self, face_img: torch.Tensor, source_latent: torch.Tensor) -> torch.Tensor:
            # Encode input face
            encoded = self.encoder(face_img)
            
            # Mix with source latent
            mixed = 0.7 * encoded + 0.3 * source_latent
            
            # Decode to swapped face
            swapped = self.decoder(mixed)
            
            return swapped
    
    return DummyFaceSwapModel()


def benchmark_model(model, test_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                   name: str, warmup_runs: int = 10) -> Dict[str, float]:
    """Бенчмарк модели."""
    print(f"🔥 Тестирование {name}...")
    
    # Прогрев
    for i in range(warmup_runs):
        face_img, source_latent = test_data[0]
        with torch.no_grad():
            _ = model(face_img, source_latent)
    
    torch.cuda.synchronize()
    
    # Основной тест
    start_time = time.time()
    
    for face_img, source_latent in test_data:
        with torch.no_grad():
            _ = model(face_img, source_latent)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(test_data) * 1000  # ms
    
    print(f"  ⏱️  Общее время: {total_time:.3f}с")
    print(f"  📊 Среднее время на кадр: {avg_time:.2f}ms")
    print(f"  🚀 FPS: {1000/avg_time:.1f}")
    
    return {
        "total_time": total_time,
        "avg_time_ms": avg_time,
        "fps": 1000/avg_time
    }


def test_torch_compile_modes():
    """Тестировать различные режимы torch.compile()."""
    print("🚀 ТЕСТИРОВАНИЕ TORCH.COMPILE() УСКОРЕНИЯ")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA недоступен, тест может быть неточным")
        return
    
    if not check_torch_compile_support():
        print("❌ torch.compile() недоступен")
        return
    
    # Создать тестовые данные
    print("📊 Подготовка тестовых данных...")
    batch_size = 4
    num_batches = 25
    test_data = []
    
    for _ in range(num_batches):
        face_img = torch.randn(batch_size, 3, 128, 128).cuda()
        source_latent = torch.randn(batch_size, 512).cuda()
        test_data.append((face_img, source_latent))
    
    print(f"  - Размер батча: {batch_size}")
    print(f"  - Количество батчей: {num_batches}")
    print(f"  - Общее количество кадров: {batch_size * num_batches}")
    
    # Создать модели
    print("\n🏗️  Подготовка моделей...")
    base_model = create_dummy_face_swap_model().cuda()
    
    # Тестировать обычную модель
    print(f"\n{'='*60}")
    base_results = benchmark_model(base_model, test_data, "Обычная модель")
    
    # Тестировать различные режимы компиляции
    compile_modes = [
        ("default", {}),
        ("reduce-overhead", {"mode": "reduce-overhead"}),
        ("reduce-overhead-simple", {"mode": "reduce-overhead", "fullgraph": False}),
    ]
    
    compile_results = {}
    
    for mode_name, compile_kwargs in compile_modes:
        print(f"\n{'='*60}")
        try:
            print(f"🔧 Компиляция модели ({mode_name})...")
            compiled_model = torch.compile(base_model, **compile_kwargs)
            
            compile_results[mode_name] = benchmark_model(
                compiled_model, test_data, f"torch.compile({mode_name})")
                
        except Exception as e:
            print(f"❌ Ошибка компиляции {mode_name}: {e}")
            continue
    
    # Сравнение результатов
    print(f"\n{'='*60}")
    print("📈 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    print(f"{'Режим':<30} {'Время (мс)':<12} {'FPS':<8} {'Ускорение':<10}")
    print("-" * 60)
    
    # Базовая модель
    base_fps = base_results["fps"]
    print(f"{'Обычная модель':<30} {base_results['avg_time_ms']:<12.2f} {base_fps:<8.1f} {'1.0x':<10}")
    
    # Скомпилированные модели
    if compile_results:
        for mode_name, results in compile_results.items():
            speedup = results["fps"] / base_fps
            speedup_color = "🚀" if speedup > 1.5 else "📈" if speedup > 1.1 else "📊"
            print(f"{mode_name:<30} {results['avg_time_ms']:<12.2f} {results['fps']:<8.1f} {speedup_color}{speedup:<9.1f}x")
    else:
        print(f"{'Компиляция недоступна':<30} {'N/A':<12} {'N/A':<8} {'N/A':<10}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if compile_results:
        best_mode = None
        best_speedup = 1.0
        
        for mode_name, results in compile_results.items():
            speedup = results["fps"] / base_fps
            if speedup > best_speedup:
                best_speedup = speedup
                best_mode = mode_name
        
        if best_mode:
            print(f"  ✅ Лучший режим: {best_mode} (ускорение {best_speedup:.1f}x)")
            
            if best_speedup > 2.0:
                print("  🎉 Отличное ускорение! Определенно стоит использовать torch.compile()")
            elif best_speedup > 1.5:
                print("  ✅ Хорошее ускорение, рекомендуется использовать")
            elif best_speedup > 1.1:
                print("  📈 Умеренное ускорение, может быть полезно")
            else:
                print("  ⚠️  Минимальное ускорение, возможно есть накладные расходы")
        else:
            print("  ❌ Не удалось скомпилировать модель")
    else:
        print("  ⚠️  torch.compile() недоступен на этой системе")
        print("  💡 Для ускорения установите Triton: pip install triton")
        print("  💡 Или используйте обычную модель - она тоже работает отлично!")
    
    # Информация о GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n🎮 Информация о GPU:")
        print(f"  - Название: {props.name}")
        print(f"  - Память: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Вычислительная способность: {props.major}.{props.minor}")
        
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  - Используется памяти: {memory_used:.1f} GB")
        print(f"  - Зарезервировано: {memory_reserved:.1f} GB")


def main():
    """Основная функция."""
    try:
        test_torch_compile_modes()
    except KeyboardInterrupt:
        print("\n❌ Тест прерван пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 