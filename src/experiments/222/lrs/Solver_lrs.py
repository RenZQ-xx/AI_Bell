import os
import stat
import subprocess
import time
from pathlib import Path

# --- 路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_DIR = PROJECT_ROOT / "external" / "lrslib-073a"
MPLRS_BIN = EXTERNAL_DIR / "mplrs"


def ensure_executable(path: Path):
    if path.exists():
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


def run_polytope_solver(input_file_path, output_file_path, num_processes=4):
    input_path = Path(input_file_path)
    output_path = Path(output_file_path)

    print(f"\n🚀 启动并行计算 (mplrs, np={num_processes})...")

    if not MPLRS_BIN.exists():
        raise FileNotFoundError(f"找不到 mplrs: {MPLRS_BIN}")
    ensure_executable(MPLRS_BIN)

    cmd = [
        "mpirun",
        "-np", str(num_processes),
        str(MPLRS_BIN),
        str(input_path),
    ]

    with open(output_path, "w") as outfile:
        process = subprocess.Popen(
            cmd,
            stdout=outfile,
            stderr=subprocess.PIPE,
            text=True,
        )

        start_time = time.time()

        try:
            while process.poll() is None:
                time.sleep(2)
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    elapsed = int(time.time() - start_time)
                    print(
                        f"\r⏳ 已运行 {elapsed}秒 | 输出文件: {size_mb:.2f} MB (正在写入...)",
                        end="",
                        flush=True,
                    )

            print(f"\n✅ 进程结束，返回码: {process.returncode}")

            stderr_out = process.stderr.read()
            if stderr_out and "error" in stderr_out.lower():
                print(f"❌ 可能存在的错误信息:\n{stderr_out}")

        except KeyboardInterrupt:
            print("\n🛑 用户手动停止计算。")
            process.terminate()
            process.wait()

    if output_path.exists():
        line_count = sum(1 for _ in open(output_path, "rb"))
        print(f"🎉 计算完成！共发现 {line_count - 5} 个面 (大致数值)。")
        print(f"📁 结果保存在: {output_path}")


if __name__ == "__main__":
    inp = PROJECT_ROOT / "data" / "polytope_222.ext"
    out = PROJECT_ROOT / "data" / "facets_222.txt"

    inp.parent.mkdir(exist_ok=True)

    if inp.exists():
        run_polytope_solver(inp, out, num_processes=4)
    else:
        print("❌ 找不到输入文件，请先生成 polytope_222.ext")
