#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Автоматический фикс для проблемы с basicsr degradations.py импортами.

Эта проблема возникает из-за изменений в torchvision, где
rgb_to_grayscale был перемещен из functional_tensor в functional.
"""

import os
import sys
import site
from pathlib import Path
from typing import List, Optional

def find_basicsr_degradations() -> List[Path]:
    """Находит все возможные пути к degradations.py в basicsr."""
    possible_paths = []
    
    # 1. Стандартные пути site-packages
    for site_path in site.getsitepackages():
        degradations_path = Path(site_path) / "basicsr" / "data" / "degradations.py"
        if degradations_path.exists():
            possible_paths.append(degradations_path)
    
    # 2. Пути для virtual environments
    if hasattr(site, 'getusersitepackages'):
        user_site = Path(site.getusersitepackages()) / "basicsr" / "data" / "degradations.py"
        if user_site.exists():
            possible_paths.append(user_site)
    
    # 3. Conda environments
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_path = Path(conda_prefix) / "Lib" / "site-packages" / "basicsr" / "data" / "degradations.py"
        if conda_path.exists():
            possible_paths.append(conda_path)
    
    # 4. Виртуальные окружения рядом с проектом
    current_dir = Path.cwd()
    for venv_name in ['venv', '.venv', 'env', '.env']:
        venv_path = current_dir / venv_name / "Lib" / "site-packages" / "basicsr" / "data" / "degradations.py"
        if venv_path.exists():
            possible_paths.append(venv_path)
        
        # Также проверяем на Linux/Mac
        venv_path_unix = current_dir / venv_name / "lib" / "python*" / "site-packages" / "basicsr" / "data" / "degradations.py"
        for python_dir in (current_dir / venv_name / "lib").glob("python*"):
            unix_path = python_dir / "site-packages" / "basicsr" / "data" / "degradations.py"
            if unix_path.exists():
                possible_paths.append(unix_path)
    
    # 5. Попытка найти через sys.path
    for sys_path in sys.path:
        if sys_path:
            degradations_path = Path(sys_path) / "basicsr" / "data" / "degradations.py"
            if degradations_path.exists():
                possible_paths.append(degradations_path)
    
    # Убираем дубликаты
    return list(set(possible_paths))

def check_degradations_file(file_path: Path) -> bool:
    """Проверяет, нужно ли исправлять файл degradations.py."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем проблемную строку
        return "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in content
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return False

def fix_degradations_file(file_path: Path) -> bool:
    """Исправляет файл degradations.py."""
    try:
        # Читаем файл
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Делаем backup
        backup_path = file_path.with_suffix('.py.backup')
        with backup_path.open('w', encoding='utf-8') as f:
            f.write(content)
        print(f"📦 Created backup: {backup_path}")
        
        # Заменяем проблемную строку
        old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new_import = "from torchvision.transforms.functional import rgb_to_grayscale"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            # Записываем исправленный файл
            with file_path.open('w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[SUCCESS] Fixed file: {file_path}")
            return True
        else:
            print(f"[WARNING] Problematic line not found in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing file {file_path}: {e}")
        return False

def verify_fix(file_path: Path) -> bool:
    """Проверяет, что фикс применен корректно."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем, что новый импорт есть и старого нет
        has_new_import = "from torchvision.transforms.functional import rgb_to_grayscale" in content
        has_old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in content
        
        return has_new_import and not has_old_import
    except Exception:
        return False

def main():
    """Main function for automatic basicsr fixing."""
    print("🔧 Automatic fix for basicsr degradations.py")
    print("=" * 50)
    
    # Search for degradations.py files
    print("🔍 Searching for basicsr degradations.py files...")
    degradations_files = find_basicsr_degradations()
    
    if not degradations_files:
        print("❌ basicsr/data/degradations.py files not found")
        print("💡 Make sure basicsr is installed:")
        print("   pip install basicsr")
        return
    
    print(f"[SUCCESS] Found {len(degradations_files)} file(s)")
    
    fixed_count = 0
    for file_path in degradations_files:
        print(f"\n[CHECK] Checking: {file_path}")
        
        if not check_degradations_file(file_path):
            print("[OK] File already fixed or doesn't contain problematic line")
            continue
        
        print("🔧 Fix required...")
        
        if fix_degradations_file(file_path):
            if verify_fix(file_path):
                print("✅ Fix applied successfully")
                fixed_count += 1
            else:
                print("❌ Error verifying fix")
        else:
            print("❌ Failed to apply fix")
    
    print("\n" + "=" * 50)
    if fixed_count > 0:
        print(f"🎉 Successfully fixed {fixed_count} file(s)")
        print("📺 Video processing should now work correctly")
    else:
        print("ℹ️ No files requiring fixes found")
    
    print("\n💡 Note: This fix is only needed for video processing")
    print("   Real-time face swap works without this fix")

if __name__ == "__main__":
    main() 