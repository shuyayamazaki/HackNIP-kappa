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


def train():
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

    # 커널 선택
    if wandb.config.kernel == "RBF":
        kernel = RBF(length_scale=wandb.config.length_scale)
    elif wandb.config.kernel == "Matern":
        kernel = Matern(length_scale=wandb.config.length_scale, nu=wandb.config.nu)
    elif wandb.config.kernel == "RationalQuadratic":
        kernel = RationalQuadratic(length_scale=wandb.config.length_scale, alpha=wandb.config.alpha)

    # Gaussian Process 모델 생성 및 학습
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 30)
    gp.fit(X_train, y_train)

    # 예측 및 성능 평가
    y_pred = gp.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 메트릭 로깅
    wandb.log({"mae": mae, "r2": r2})


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",  # Bayesian Optimization
        "metric": {"name": "mae", "goal": "minimize"},
        "parameters": {
            "kernel": {"values": ["RBF", "Matern"]},
            "nu": {"values": [0.5, 1.0, 1.5]},  # Matern 전용
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="orb-gp-hyperparam-optimization")

    wandb.agent(sweep_id, train, count=30)
