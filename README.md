# ion_conductivity   
**Purpose:** Develop a ionic conductivity prediction model $\rightarrow$ Predict amorphous structure from crystalline structure, temperatrue

## Introduction
Li ion battery의 conversion cathode 에서 amourphous구조는 crystalline보다 reversibility가 좋을 수도 있다.   
MP에서 Li포함된 개수가 22000   
이들 중 band gap으로 3eV cutoff으로 이온 interaction 하는 애들 거르면 1.5k   
amorphous structure를 PFP로 우선 구함   
계산 도중 오류가 날 때가 있는데 1.5k보다는 적다   
온도를 5개 찍기 때문에 (1000, 1500, 2000, 2500, 5000) 적어도 5k개....   
각 amorphous구조를 NVT로 파라미터 지정해서 ionic conductivity 계산   
개수가 충분하지 않으니 [SOAP](https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html)사용 고려 vs GNN   
현재MPContribut에서는 amorphous crystal 개수가 서로 달라서 결정/비정질 중 어떤 걸 입력으로 쓰는게 좋은지 비교가 어려움   
MPContribut에서 RDF, Diffusivity계산했는데 거의 똑같았음.   
1.5k 다 완성되면 둘 중 어느게 더 예측에 유리한지 확인   

-----------
## Installation
```
conda create -n ionic python=3.10.13
conda activate ionic
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
python -m pip install lightning
pip install hydra-core joblib wandb matplotlib scikit-learn python-dotenv jarvis-tools pymatgen ase rdkit tqdm transformers datasets diffusers fairchem-core

```

## Pipeline
### Data
- 현재 가진 데이터 여기에 정리해주세요 - Resources란 UnlabeledDataset_v2-2
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
- [x] arena에서 PFP 사용이 가능하고
- [x] amorphous구조 하나 만드는데 얼마나 걸리는지 (60/day without parallel computing)
- [x] crystalline CIF만들어서 column 따로 추가 (링크는 아래 Resources/Amorphous diffusivity/data with crystalline structures in csv)
- [x] MPContrib 중에 MP-ID 없는 애들도 crystal CIF, amorphous CIF 동시에 정리해두기 - 없는 애들은 없음 ㅠ
- [x] 입력 amorphous/crystalline 중 어떤게 더 잘 맞는지 평가 - 그냥 amorphous 가기로 함
- [x] SOAP 사용가능 여부 확인 (위에꺼랑 같이 총 4개 실험. 해봤는데 결과가 예측이 잘 안되면 여기서부터 다시시작) - 메모리너무많이 차지해서 폐기
- [x] 논문의 일부 feature를 이용하여 RF구현-만족할만한 결과가 나오지 않았음. 그리고 설명도 불충분하고 코드도 없어서 이걸 따라가는 건 폐기. 대신 GNN으로 이전에 얻은 결과가 나쁘진 않아서 GNN으로 가자
- [ ] 배터리를 실제 충방전하는 경우 Stochiometry가 Li 100% 가 아니게 됨. 이것도 미래에 고려할 것이지만 우선은 100%만 먼저 고려해보자.
- [ ] GNN+AL 현재 가진 unlabled, labeled data에서 훈련 후 경과 보기
- [ ] 현재 가진 데이터 갯수 정리

## Logic
### Purpose
서로다른 구조에 대해 예측

### Construct unlabled dataset
- Li ion conductivity 로 제한 (MP에서) (<10k)
- 
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
- [Unlabeled dataset_v2-2](https://drive.google.com/file/d/13WIXvU-JAk1IxEu58DT7uotmUYHHekrz/view?usp=sharing): Materials Project database에서 Li-containing compounds 중 bandgap > 3 eV 이상인 chemical composition 들의 amorphous structure; structure_cif열이 비어있지 않은 행만 사용 -> Dataset size = 5916
