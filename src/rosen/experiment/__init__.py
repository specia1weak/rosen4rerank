from pathlib import Path
from .config import ROOT_DIR, WORKSPACE

def ignore_future_warning():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

def change_root_workdir():
    import os
    print("切换到: ", ROOT_DIR)
    os.chdir(ROOT_DIR)

def set_all():
    ignore_future_warning()
    change_root_workdir()

if __name__ == '__main__':
    print(ROOT_DIR)