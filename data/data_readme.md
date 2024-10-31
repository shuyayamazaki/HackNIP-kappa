# How to prepare dataset

The original data should contain `cif` column.   
`cif`가 포함된 데이터는 graph data로 변환되어 `.parquet`으로 저장.   
Split 방법도 정한 이후, 이름을 공유하는 `_train.parquet`, `_val.parquet`, and `_test.parquet`으로 저장.
graph data와 label이 포함된 데이터로부터 `dataloader`를 만듬.   

