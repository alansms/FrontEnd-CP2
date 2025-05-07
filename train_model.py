import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Exemplo de estrutura do CSV:
# idade,renda,gasto_mensal,tempo_como_cliente,classe
# 30,5000,800,12,1

project_dir = os.path.dirname(__file__)
data_path = os.path.join(project_dir, "data", "exemplo_batch.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"O arquivo de dados não foi encontrado: {data_path}")

models_dir = os.path.join(project_dir, "models")
# Limpa a pasta models antes de criar o novo modelo
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)
os.makedirs(models_dir, exist_ok=True)

# 1) Carrega dados de exemplo
df = pd.read_csv(data_path)

# 2) Define X e y
features = ['idade', 'renda', 'gasto_mensal', 'tempo_como_cliente']
X = df[features]
y = df['classe']

# 3) Divide treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Treina modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5) Avalia
acc = model.score(X_test, y_test)
print(f"Acurácia no teste: {acc:.2%}")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:\n", cm)

report = classification_report(y_test, y_pred, target_names=["0 - Não Propenso", "1 - Propenso"])
print("\nRelatório de Classificação:\n", report)

# 6) Salva .pkl
model_path = os.path.join(models_dir, "modelo_treinado.pkl")
joblib.dump(model, model_path)
print(f"Modelo salvo em {model_path}")
