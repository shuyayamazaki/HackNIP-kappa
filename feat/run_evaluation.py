import subprocess
from joblib import Parallel, delayed
import glob
import argparse

def run_evaluation(script_file, device, data_file, task_type, split_type, ml_type):
    """train_eval.py 스크립트를 실행하는 함수"""
    cmd = [
        "python", script_file,
        "--device", device,
        "--data_path", data_file,
        "--task_type", task_type,
        "--split_type", split_type,
        "--ml_type", ml_type,
    ]
    
    print(f"Running command: {' '.join(cmd)}")  # 실행되는 명령어 출력
    process = subprocess.Popen(cmd)  # 비동기 실행 (백그라운드 실행)
    process.wait()  # 프로세스 완료 대기

def main(args):    
    # Script 파일
    mlip_type = args.mlip_type
    ml_type = args.ml_type
    SCRIPT_FILE = f"train_eval_{mlip_type}.py"
    # 실행할 장치 및 데이터 파일 목록
    # DEVICES = ["cuda:0", 
    #            "cuda:0", 
    #            "cuda:0", 
    #            "cuda:0", 
    #            "cuda:0", 
    #            "cuda:0", 
    #            "cuda:0", 
    #            "cuda:0"
    #            ]
    # DATA_FILES = [
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/BACE_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/BBBP_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/ClinTox_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/ESOL_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/FreeSolv_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/HIV_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/Lipophilicity_dataset_relaxed.pkl",
    #     "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/SIDER_dataset_relaxed.pkl",
    # ]
    # TASK_TYPE = [ml
    #     'classification',
    #     'classification',
    #     'classification',
    #     'regression',
    #     'regression',
    #     'classification',
    #     'regression',
    #     'classification',
    # ]
    # SPLIT_TYPE = [
    #     'scaffold',
    #     'scaffold',
    #     'random',
    #     'random',
    #     'random',
    #     'scaffold',
    #     'random',
    #     'random',
    # ]

    preprocessed_dir = "/home/sokim/ion_conductivity/feat/preprocessed_data"

    # Find and sort all *_relaxed.pkl files
    # DATA_FILES = sorted(glob.glob(f"{preprocessed_dir}/*_relaxed.pkl"))
    DATA_FILES = ["/home/sokim/ion_conductivity/feat/preprocessed_data/supercon_relaxed.pkl"]

    # 병렬 실행
    n_jobs = len(DATA_FILES)
    n_jobs = 3
    Parallel(n_jobs=n_jobs)(
        delayed(run_evaluation)(SCRIPT_FILE, "cuda:1", DATA_FILES[i], 'regression', 'random', ml_type)
        # delayed(run_evaluation)(SCRIPT_FILE, DEVICES[i], DATA_FILES[i], TASK_TYPE[i], SPLIT_TYPE[i], ML_TYPE[i])
        for i in range(len(DATA_FILES))
    ) 

    print("All scripts have finished running.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mlip_type', type=str, default='orb', help='MLIP model to extract features from (orb, eqV2, or mace)')
    parser.add_argument('--ml_type', type=str, default='mlp', help='Type of ML model (mlp or xgb)')
    args = parser.parse_args()
    main(args)
