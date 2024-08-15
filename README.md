# ion_conductivity   
**Purpose:** Develop a ionic conductivity prediction model $\rightarrow$ Predict amorphous structure from crystalline structure, temperatrue

## Introduction
Li ion battery의 conversion cathode 에서 amourphous구조는 crystalline보다 reversibility가 좋을 수도 있다.


-----------
## Installation
```
conda create -n ionic python=3.10.13
```
## Pipeline
- Preprocessing: Datasource $\rightarrow$ data_save
  - 데이터 전처리 후 POSCAR로 디렉토리에 저장
  - 각 POSCAR와 ionic conductivity label 지정 (condition to specify: ion of interest (Li), temperature (All))
    - Pandas DataFrame column1: poscar column2: label
  - 용량이 총합 50MB넘어가는지 여부 확인
- DataLoader: data_save  $\rightarrow$ graph structure
- Model: graph structure $\rightarrow$ prediction

-------------------
## TODO
- [x] Preprocessing pipeline
- [x] PFP 사용해도 좋음 (Prof.Li)
- [x] PFP로 Diffusivity 계산 (4.5h)
- [x] PFP로 non-stoichiometric한 ionic conductivity계산이 맞는지
- [x] amorphous구조 하나 만드는데 얼마나 걸리는지 (60/day without parallel computing)
- [ ] 입력 amorphous 대신 crystalline 넣어서 예측
- [x] crystalline CIF만들어서 column 따로 추가 (링크는 아래 Resources/Amorphous diffusivity/data with crystalline structures in csv)
- [ ] 있는 거로 diffusion먼저

## Logic
### Purpose
서로다른 구조에 대해 예측

### Construct unlabled dataset
- Li ion conductivity 로 제한 (MP에서) (<10k)
- Non-stoichiometric 으로 확장

## Diffusion for amorphous
- Fix adjacency matrix, only denoise coordinates


## Resources
- [Amorpous diffusivity](https://contribs.materialsproject.org/projects/amorphous_diffusivity);
  - [data in csv](https://drive.google.com/file/d/1KZn4WD3NLvlD1lr4PGvCBqZ80Syk5Vzr/view?usp=sharing)
  - [data with crystalline structures in csv](https://drive.google.com/file/d/1-2YsXG4ezZaHTZsnm3l2swgVw0LO7kDI/view?usp=sharing)
- [MP contribs download](https://docs.materialsproject.org/downloading-data/query-and-download-contributed-data)
- [The ab initio amorphous materials database: Empowering machine learning to decode diffusivity](https://ar5iv.labs.arxiv.org/html/2402.00177)
- [MACE](https://github.com/ACEsuit/mace?tab=readme-ov-file): 만약에 그냥 신경망 성능이 맘에 안들면 해보는 걸로
