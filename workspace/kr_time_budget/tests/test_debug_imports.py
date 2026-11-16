"""IDE先初始化调试器，再执行脚本；入口目录不能遮蔽标准库code。"""
from pathlib import Path
import subprocess
import sys


def test_debug_entry_directory_preserves_stdlib_console():
    workspace = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-c',
         'from code import InteractiveConsole, InteractiveInterpreter, compile_command; '
         'assert compile_command("1 + 1") is not None'],
        cwd=workspace, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
