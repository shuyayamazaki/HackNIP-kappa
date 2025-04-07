


## 1. Run preprocessing codes
Codes start with `preprocessing_relaxation`
아래 훈련 및 평가 과정을 동일한 코드로 수행 가능하도록 서로 다른 데이터 소스에서 `.pkl`형식으로 통일하는 작업. 
- 현재 구현된 코드
    - [x] `preprocessing_relaxation_bandgap.py`
    - [x] `preprocessing_relaxation_diffusivity.py`
    - [x] `preprocessing_relaxation_moleculenet.py`
    - [x] `preprocessing_relaxation_mptrj.py`
    - [ ] `preprocessing_relaxation_pnas.py`
    - [ ] `preprocessing_relaxation_supercon.py`

## 2. Train and evaluate ML models
Codes start with `run_eval`
- 현재 구현된 코드
    - [x] `train_eval_eqV2.py`
    - [x] `train_eval_orb.py`
    - [ ] `train_eval_mace.py`
    - [ ] `train_eval_featurizer.py`
    
## 3. Auto-execution
Codes start with `run_`