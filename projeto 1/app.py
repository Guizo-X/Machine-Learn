import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Carregue o conjunto de dados a partir do arquivo CSV
data = pd.read_csv('C:\\Users\\Guizo\\Downloads\\iris\\iris.data')

# Especifica a matriz de atributos (X) e o vetor de classes (y)
X = data.drop('Iris-setosa', axis=1)
y = data['Iris-setosa']

# Divide o conjunto de dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crie um modelo SVM
model = SVC()

# Treine o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Faça previsões com o conjunto de teste
y_pred = model.predict(X_test)

# Calcule a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Exiba um relatório de classificação
print(classification_report(y_test, y_pred))

# Exiba a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(cm)

# Crie um objeto de scaler
scaler = StandardScaler()

# Ajuste o scaler aos dados de treinamento e, em seguida, aplique a transformação
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Defina a grade de hiperparâmetros
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# Crie um classificador de grade
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

# Execute a pesquisa em grade
grid_search.fit(X_train_std, y_train)

# Obtenha os melhores hiperparâmetros
best_params = grid_search.best_params_
print("Melhores Hiperparâmetros:", best_params)

# Avalie o modelo com os melhores hiperparâmetros
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia com Melhores Hiperparâmetros: {accuracy:.2f}')

# Identifique a classe com o maior valor de dados
class_counts = y.value_counts()
most_frequent_class = class_counts.idxmax()

# Exiba a classe com o maior valor
print(f'Classe com Maior Valor de Dados: {most_frequent_class}')

# Crie um gráfico de barras mostrando a distribuição de classes com cores diferentes
class_counts = y.value_counts()
class_distribution = class_counts.plot(kind='bar', color=['blue', 'red', 'yellow'])
plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.show()

# Crie a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Exiba a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()

# Identifique a classe com o maior valor de dados
class_counts = y.value_counts()
most_frequent_class = class_counts.idxmax()

# Exiba a classe com o maior valor
print(f'Classe mais frequente: {most_frequent_class}')

# Crie um gráfico de barras mostrando a distribuição de classes com cores diferentes
class_counts = y.value_counts()
colors = ['yellow' if cls == most_frequent_class else 'red' for cls in class_counts.index]
class_distribution = class_counts.plot(kind='bar', color=colors)
plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Contagem')

# Adicione uma legenda ao gráfico de barras
most_frequent_patch = mpatches.Patch(color='yellow', label='Classe mais frequente')
others_patch = mpatches.Patch(color='red', label='Outras Classes')
plt.legend(handles=[most_frequent_patch, others_patch])

plt.show()


