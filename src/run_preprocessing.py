import subprocess
from joblib import Parallel, delayed

def run_preprocessing(device, data_file, property_cols):
    """preprocessing_relaxation.py 스크립트를 실행하는 함수"""
    cmd = [
        "python", "preprocessing_relaxation.py",
        "--device", device,
        "--data_path", data_file,
        "--property_cols", property_cols,
    ]
    
    print(f"Running command: {' '.join(cmd)}")  # 실행되는 명령어 출력
    process = subprocess.Popen(cmd)  # 비동기 실행 (백그라운드 실행)
    process.wait()  # 프로세스 완료 대기

def main():    
    # 실행할 장치 및 데이터 파일 목록
    DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"]
    DATA_FILES = [
        "/home/lucky/Projects/llacha/data/data/ClinTox_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/ESOL_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/BACE_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/BBBP_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/FreeSolv_dataset.csv",

    ]
    PROPERTY_COLS = [
        '["FDA_APPROVED", "CT_TOX"]',
        '["measured log solubility in mols per litre"]',
        '["Class"]',
        '["p_np"]',
        '["expt"]',
    ]

    # 병렬 실행
    Parallel(n_jobs=len(DEVICES))(
        delayed(run_preprocessing)(DEVICES[i], DATA_FILES[i], PROPERTY_COLS[i])
        for i in range(len(DEVICES))
    ) 

    # 실행할 장치 및 데이터 파일 목록
    DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    DATA_FILES = [
        "/home/lucky/Projects/llacha/data/data/Tox21_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/Lipophilicity_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/HIV_dataset.csv",
        "/home/lucky/Projects/llacha/data/data/SIDER_dataset.csv",

    ]
    PROPERTY_COLS = [
        '["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]',
        '["exp"]',
        '["HIV_active"]',
        '["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", "Reproductive system and breast disorders", "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", "General disorders and administration site conditions", "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders", "Injury, poisoning and procedural complications"]',
    ]

    # 병렬 실행
    Parallel(n_jobs=len(DEVICES))(
        delayed(run_preprocessing)(DEVICES[i], DATA_FILES[i], PROPERTY_COLS[i])
        for i in range(len(DEVICES))
    )

    print("All scripts have finished running.")


if __name__ == "__main__":
    main()
