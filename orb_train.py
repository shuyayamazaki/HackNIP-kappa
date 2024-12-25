import pickle
import pandas as pd
import numpy as np
# Gaussian process regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

# open graph_features.pkl
with open('graph_features.pkl', 'rb') as f:
    X = pickle.load(f)

df = pd.read_parquet('/home/lucky/Projects/ion_conductivity/data/MPContribs_armorphous_diffusivity.parquet')
df = df[df['data_properties_A_element'] == 'Li']  # only Li-ion conductivity
df = df[df['data_temperature_value']<5000]  # Remove 5000 K data
df['y'] = df['data_properties_A_diffusivity_value'].apply(np.log10)

y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# WandB 설정
wandb.init(project="gp-hyperparam-optimization")

# 하이퍼파라미터 가져오기
config = wandb.config

# 커널 선택
if config.kernel == "RBF":
    kernel = RBF(length_scale=config.length_scale)
elif config.kernel == "Matern":
    kernel = Matern(length_scale=config.length_scale, nu=config.nu)
elif config.kernel == "RationalQuadratic":
    kernel = RationalQuadratic(length_scale=config.length_scale, alpha=config.alpha)

# Gaussian Process 모델 생성 및 학습
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = gp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 메트릭 로깅
wandb.log({"mae": mae, "r2": r2})