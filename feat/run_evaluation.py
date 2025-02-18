import subprocess
from joblib import Parallel, delayed

def run_evaluation(script_file, device, data_file, task_type, split_type):
    """train_eval.py 스크립트를 실행하는 함수"""
    cmd = [
        "python", script_file,
        "--device", device,
        "--data_path", data_file,
        "--task_type", task_type,
        "--split_type", split_type,
    ]
    
    print(f"Running command: {' '.join(cmd)}")  # 실행되는 명령어 출력
    process = subprocess.Popen(cmd)  # 비동기 실행 (백그라운드 실행)
    process.wait()  # 프로세스 완료 대기

def main():    
    # Script 파일
    SCRIPT_FILE = "train_eval_eqV2.py"
    # 실행할 장치 및 데이터 파일 목록
    DEVICES = ["cuda:0", 
               "cuda:0", 
               "cuda:0", 
               "cuda:0", 
               "cuda:0", 
               "cuda:0", 
               "cuda:0", 
               "cuda:0"
               ]
    DATA_FILES = [
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/BACE_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/BBBP_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/ClinTox_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/ESOL_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/FreeSolv_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/HIV_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/Lipophilicity_dataset_relaxed.pkl",
        "/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/SIDER_dataset_relaxed.pkl",
    ]
    TASK_TYPE = [
        'classification',
        'classification',
        'classification',
        'regression',
        'regression',
        'classification',
        'regression',
        'classification',
    ]
    SPLIT_TYPE = [
        'scaffold',
        'scaffold',
        'random',
        'random',
        'random',
        'scaffold',
        'random',
        'random',
    ]

    # 병렬 실행
    n_jobs = len(DEVICES)
    n_jobs = 3
    Parallel(n_jobs=n_jobs)(
        delayed(run_evaluation)(SCRIPT_FILE, DEVICES[i], DATA_FILES[i], TASK_TYPE[i], SPLIT_TYPE[i])
        for i in range(len(DEVICES))
    ) 

    print("All scripts have finished running.")


if __name__ == "__main__":
    main()
