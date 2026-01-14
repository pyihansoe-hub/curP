import os

name = input("New currency (e.g., MMK-200): ")
folder_path = f'datasets/{name}'
os.makedirs(folder_path, exist_ok=True)
print(f"Created: {folder_path}")
print(f"Add images then run: python train.py")