import itertools
import csv
import json
import subprocess
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
"""
原理：只需要指定目标python文件路径就行，只要你使用了ParamManager这套，命令行传参都是第一优先级
"""

class GridSearchEngine:
    def __init__(self, script_path: str,):
        """
        :param script_path: 运行python的cwd下面找script_path相对路径
        """
        self.script_path = script_path
        self.base_python = sys.executable

    def _default_run_name(self) -> str:
        script_name = os.path.splitext(os.path.basename(self.script_path))[0]
        return script_name or "grid_search"

    def _make_run_dir(self, log_dir: str, run_name: str = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = run_name or self._default_run_name()
        run_dir = os.path.join(log_dir, f"{timestamp}_{run_label}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _log(self, engine_log_path: str, message: str):
        print(message)
        with open(engine_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    def make_grid(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """将参数空间字典展开为笛卡尔积列表"""
        keys = param_space.keys()
        values = param_space.values()
        combinations = itertools.product(*values)
        return [dict(zip(keys, combo)) for combo in combinations]

    def build_experiment_name(self, params: Dict[str, Any]) -> str:
        """根据当前参数组合动态生成实验名"""
        if "_name" in params:
            name = str(params["_name"])
            if "seed" in params:
                name = f"{name}_seed{params['seed']}"
            return name

        parts = []
        for k, v in params.items():
            if k == "experiment_name" or k.startswith("_"):
                continue
            # 简化浮点数等长格式
            val_str = str(v).replace(".", "_") if isinstance(v, float) else str(v)
            parts.append(f"{k}{val_str}")
        if parts:
            return "-".join(parts)
        if "experiment_name" in params:
            return str(params["experiment_name"])
        return "baseline"

    def worker(self, params: Dict[str, Any], gpu_id: int, log_dir: str):
        """单个实验的执行工作流"""
        # 1. 构建实验名并注入参数
        run_params = dict(params)
        exp_name = self.build_experiment_name(run_params)
        if "experiment_name" not in run_params:
            run_params["experiment_name"] = exp_name
        cmd = [self.base_python, self.script_path]
        for k, v in run_params.items():
            if k.startswith("_"):
                continue
            cmd.extend([f"--{k}", str(v)])
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{exp_name}.log")
        start_time = time.time()
        return_code = None
        error_message = ""

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
            return_code = int(process.returncode)
            status = "success" if process.returncode == 0 else "failed"
        except Exception as e:
            status = "error"
            error_message = str(e)

        end_time = time.time()
        return {
            "exp_name": exp_name,
            "status": status,
            "gpu_id": gpu_id,
            "log_path": log_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "return_code": return_code,
            "params": run_params,
            "error_message": error_message,
        }

    def run(
            self,
            param_space: Dict[str, List[Any]],
            available_gpus: List[int],
            log_dir: str = "./logs",
            run_name: str = None,
    ):
        tasks = self.make_grid(param_space)
        run_dir = self._make_run_dir(log_dir=log_dir, run_name=run_name)
        experiment_log_dir = os.path.join(run_dir, "experiments")
        os.makedirs(experiment_log_dir, exist_ok=True)
        engine_log_path = os.path.join(run_dir, "engine.log")
        summary_path = os.path.join(run_dir, "summary.csv")

        self._log(engine_log_path, f"📁 RunDir: {run_dir}")
        self._log(engine_log_path, f"📊 发现 {len(tasks)} 个实验组合，可用 GPU: {available_gpus}")
        self._log(engine_log_path, f"🧩 ParamSpace: {json.dumps(param_space, ensure_ascii=False, sort_keys=True)}")
        max_workers = len(available_gpus) if available_gpus else 1

        with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(
                summary_file,
                fieldnames=[
                    "exp_name",
                    "status",
                    "gpu_id",
                    "return_code",
                    "start_time",
                    "end_time",
                    "duration",
                    "log_path",
                    "params",
                    "error_message",
                ],
            )
            writer.writeheader()

            future_to_meta = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, task_params in enumerate(tasks):
                    # 轮询分配 GPU
                    gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
                    exp_name = self.build_experiment_name(task_params)
                    log_path = os.path.join(experiment_log_dir, f"{exp_name}.log")
                    self._log(
                        engine_log_path,
                        f"🚀 [Start] GPU:{gpu_id} | Exp:{exp_name} | Log:{log_path}",
                    )
                    future = executor.submit(self.worker, task_params, gpu_id, experiment_log_dir)
                    future_to_meta[future] = {"exp_name": exp_name, "gpu_id": gpu_id, "log_path": log_path}

                for future in as_completed(future_to_meta):
                    meta = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "exp_name": meta["exp_name"],
                            "status": "error",
                            "gpu_id": meta["gpu_id"],
                            "return_code": None,
                            "start_time": None,
                            "end_time": None,
                            "duration": None,
                            "log_path": meta["log_path"],
                            "params": {},
                            "error_message": str(e),
                        }

                    writer.writerow(
                        {
                            "exp_name": result["exp_name"],
                            "status": result["status"],
                            "gpu_id": result["gpu_id"],
                            "return_code": result["return_code"],
                            "start_time": result["start_time"],
                            "end_time": result["end_time"],
                            "duration": result["duration"],
                            "log_path": result["log_path"],
                            "params": json.dumps(result["params"], ensure_ascii=False, sort_keys=True),
                            "error_message": result["error_message"],
                        }
                    )
                    summary_file.flush()

                    status_icon = {"success": "✅ Success", "failed": "❌ Failed", "error": "💥 Error"}.get(
                        result["status"], result["status"]
                    )
                    duration = result["duration"]
                    duration_text = f"{duration:.1f}s" if duration is not None else "unknown"
                    self._log(
                        engine_log_path,
                        f"{status_icon} [{duration_text}] | Exp:{result['exp_name']} | Log:{result['log_path']}",
                    )

        self._log(engine_log_path, f"🧾 Summary: {summary_path}")


if __name__ == '__main__':
    # 定义搜索空间 (列表中包含你需要尝试的各种值)
    search_space = {
        "model": ["star", "ple"],
        "seed": [2024, 2025, 2026],
        "device": ["cuda"]
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="run.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)
