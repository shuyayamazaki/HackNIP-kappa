# ion_conductivity   
**Purpose:** Develop a ionic conductivity prediction model


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
- [ ] Preprocessing pipeline
- [ ] GNOME availability

## Resources
- [Amorpous diffusivity](https://contribs.materialsproject.org/projects/amorphous_diffusivity);
  - [data in csv](https://drive.google.com/file/d/1KZn4WD3NLvlD1lr4PGvCBqZ80Syk5Vzr/view?usp=sharing)
- [MP contribs download](https://docs.materialsproject.org/downloading-data/query-and-download-contributed-data)
- [The ab initio amorphous materials database: Empowering machine learning to decode diffusivity](https://ar5iv.labs.arxiv.org/html/2402.00177)
- [MACE](https://github.com/ACEsuit/mace?tab=readme-ov-file): 만약에 그냥 신경망 성능이 맘에 안들면 해보는 걸로
