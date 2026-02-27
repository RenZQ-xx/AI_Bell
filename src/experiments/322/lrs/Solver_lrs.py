import subprocess
import shutil
import time
import os
import stat
from pathlib import Path

# --- 路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_DIR = PROJECT_ROOT / "external" / "lrslib-073a" # 请根据实际文件夹名调整
LRS_BIN = EXTERNAL_DIR / "lrs"
MPLRS_BIN = EXTERNAL_DIR / "mplrs"

def ensure_executable(path):
    """确保文件有执行权限"""
    if path.exists():
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)

def estimate_complexity(input_file):
    """
    第一步：使用 lrs -est 估算任务难度
    """
    print(f"\n🔮 [Step 1] 正在估算任务复杂度 (使用 lrs -est)...")
    
    if not LRS_BIN.exists():
        print(f"⚠️ 找不到 lrs ({LRS_BIN})，跳过估算步骤。")
        return

    ensure_executable(LRS_BIN)
    
    # 构造命令: ./lrs file.ext -est
    cmd = [str(LRS_BIN), str(input_file), "-est"]
    
    try:
        # lrs -est 通常很快，我们设置 30秒超时防止意外
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # 解析输出
        output = result.stdout
        print("-" * 40)
        found_est = False
        for line in output.splitlines():
            if "Estimated size" in line or "bases" in line:
                print(f"📊 {line.strip()}")
                found_est = True
        
        if not found_est:
            print("ℹ️ 未能解析出估算值，原始输出如下：")
            print(output[:200] + "...")
        print("-" * 40)
        
    except subprocess.TimeoutExpired:
        print("⚠️ 估算超时，任务可能非常巨大。")
    except Exception as e:
        print(f"⚠️ 估算出错: {e}")

def run_polytope_solver(input_file_path, output_file_path, num_processes=4):
    """
    第二步：并行求解并监控进度
    """
    input_path = Path(input_file_path)
    output_path = Path(output_file_path)

    # 1. 先进行估算
    # estimate_complexity(input_path)

    # 2. 准备运行 mplrs
    print(f"\n🚀 [Step 2] 启动并行计算 (mplrs, np={num_processes})...")
    
    if not MPLRS_BIN.exists():
        raise FileNotFoundError(f"找不到 mplrs: {MPLRS_BIN}")
    ensure_executable(MPLRS_BIN)

    # 构造 MPI 命令
    cmd = [
        "mpirun", 
        "-np", str(num_processes), 
        str(MPLRS_BIN), 
        str(input_path)
    ]

    # 打开文件准备写入
    with open(output_path, "w") as outfile:
        # 使用 Popen 而不是 run，这样我们可以非阻塞地监控
        process = subprocess.Popen(
            cmd,
            stdout=outfile,      # 结果直接写入文件
            stderr=subprocess.PIPE, # 捕获错误日志
            text=True
        )
        
        start_time = time.time()
        
        try:
            # 循环监控，直到进程结束
            while process.poll() is None:
                time.sleep(2) # 每2秒检查一次
                
                # 检查输出文件大小/行数
                if output_path.exists():
                    # 简单估算：读取文件大小或行数
                    # 为了不影响性能，只看文件大小 (bytes)
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    # 如果你想看具体的面数，可以用 wc -l (Linux)
                    # 或者是简单估算：通常一行大概 100-200 bytes
                    # lines_est = int(output_path.stat().st_size / 150) 
                    
                    elapsed = int(time.time() - start_time)
                    print(f"\r⏳ 已运行 {elapsed}秒 | 输出文件: {size_mb:.2f} MB (正在写入...)", end="", flush=True)

            print(f"\n✅ 进程结束，返回码: {process.returncode}")
            
            # 检查是否有错误输出
            stderr_out = process.stderr.read()
            if stderr_out and "error" in stderr_out.lower():
                print(f"❌ 可能存在的错误信息:\n{stderr_out}")

        except KeyboardInterrupt:
            print("\n🛑 用户手动停止计算。")
            process.terminate()
            process.wait()

    # 最后统计一下结果
    if output_path.exists():
        line_count = sum(1 for _ in open(output_path, 'rb')) # 快速行数统计
        print(f"🎉 计算完成！共发现 {line_count - 5} 个面 (大致数值)。") # 减去头部的元数据行
        print(f"📁 结果保存在: {output_path}")

# --- 运行入口 ---
if __name__ == "__main__":
    inp = PROJECT_ROOT / "data" / "polytope_322.ext"
    out = PROJECT_ROOT / "data" / "facets_322.txt"
    
    inp.parent.mkdir(exist_ok=True)
    
    if inp.exists():
        # 记得这里要是 >= 2
        run_polytope_solver(inp, out, num_processes=4)
    else:
        print("❌ 找不到输入文件，请先生成 polytope_322.ext")
