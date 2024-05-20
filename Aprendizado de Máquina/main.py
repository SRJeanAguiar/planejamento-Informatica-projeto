import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# Carregar cada planilha em um DataFrame
produtos_df = pd.read_excel('produtos.xlsx')
clientes_df = pd.read_excel('clientes.xlsx')
pedidos_df = pd.read_excel('pedidos.xlsx')
itens_pedido_df = pd.read_excel('itens_pedido.xlsx')
estoque_df = pd.read_excel('estoque.xlsx')
categorias_df = pd.read_excel('categorias.xlsx')
fornecedores_df = pd.read_excel('fornecedores.xlsx')

# Tratar valores ausentes, outliers e dados inconsistentes
produtos_df = produtos_df.dropna()  # Remover linhas com valores ausentes

# Confirmar a remoção de linhas com valores ausentes
print("Produtos após remover valores ausentes:")
print(produtos_df)

# Padronizar formatos e unidades
produtos_df['preço'] = produtos_df['preço'].astype(float)  # Converter preços para o tipo float


# Confirmar a conversão de formato
print("\nFormato dos preços após conversão:")
print(produtos_df['preço'].dtype)


# Exibir as primeiras linhas de cada DataFrame
print("Produtos:")
print(produtos_df)

print("\nClientes:")
print(clientes_df)

print("\nPedidos:")
print(pedidos_df)

print("\nItens do Pedido:")
print(itens_pedido_df)

print("\nEstoque:")
print(estoque_df)

print("\nCategorias:")
print(categorias_df)

print("\nFornecedores:")
print(fornecedores_df)


# Classificação dos produtos com base no preço
produtos_df['classificação'] = produtos_df['preço'].apply(lambda x: 1 if x > 50 else 0)

# Valores verdadeiros (preços reais) e valores previstos (classificação baseada no preço)
y_true = produtos_df['classificação']
y_pred = produtos_df['classificação']  # Para este exemplo, as previsões são iguais à classificação baseada no preço

# Calcula a matriz de confusão
matriz_confusao = confusion_matrix(y_true, y_pred)

# Exibe a matriz de confusão
print("Matriz de Confusão:")
print(matriz_confusao)


# Dividir os dados em conjunto de treinamento e teste
X = produtos_df[['preço']]  # Features
y = produtos_df['classificação']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escolher algoritmo de classificação
modelo = RandomForestClassifier()

# Treinar o modelo
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_test, y_pred)
print("Precisão do modelo:", precisao)

# Identifica corretamente todos os exemplos positivos.
recall = recall_score(y_test, y_pred)
print("Recall do modelo:", recall)

# Definir os parâmetros para otimização
parametros = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}

# Criar o objeto GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), parametros, cv=3)

# Executar a busca em grade
grid_search.fit(X_train, y_train)

# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Avaliar o modelo com os melhores hiperparâmetros
melhor_modelo = grid_search.best_estimator_
y_pred = melhor_modelo.predict(X_test)

# Realizar validação cruzada
pontuacoes = cross_val_score(melhor_modelo, X, y, cv=3)

# Exibir as pontuações de validação cruzada
print("Pontuações de validação cruzada:", pontuacoes)

# Calcular a média das pontuações
media_pontuacoes = pontuacoes.mean()
print("Média das pontuações de validação cruzada:", media_pontuacoes)
