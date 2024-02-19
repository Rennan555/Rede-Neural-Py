import numpy as np
import sys

##--------------------------------------------------------------------------------------------------
##----------------------------------- REDE NEURAL DE 2 CAMADAS -------------------------------------
##--------------------------------------------------------------------------------------------------

## Inicializa pesos e vieses aleatoriamente, um de cada para cada camada(layer)
def inicializar_parametros(input:int, output:int, hidden:int):

    peso1 = np.random.randn(input, hidden)
    peso2 = np.random.randn(hidden, output)

    vies1 = np.zeros((1, hidden))
    vies2 = np.zeros((1, output))

    return peso1, vies1, peso2, vies2

def sigmoid(x:np.ndarray): return 1 / (1 + np.exp(-x)) ## Função diferenciável
def sigmoid_derivada(x:np.ndarray): return x * (1 - x) ## Nunca negativa

## Diferença entre o valor real e o previsto
def calcular_erro(y_real:np.ndarray, y_prev:np.ndarray): return y_real - y_prev

## Ativas as funções e retorna a saída das camadas hidden e output
def propagacao_avancar(x:np.ndarray, peso1:np.ndarray, vies1:np.ndarray, peso2:np.ndarray, vies2:np.ndarray):

    hidden_layer_input = np.dot(x, peso1) + vies1
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, peso2) + vies2
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

## Usado para aprendizagem supervisionada, através dos gradientes de cada camada
def propagacao_retroceder(x:np.ndarray, y:np.ndarray, hidden_layer_output:np.ndarray, output_layer_output:np.ndarray, peso1:np.ndarray, vies1:np.ndarray, peso2:np.ndarray, vies2:np.ndarray, aprendizado_taxa:float):

    error = calcular_erro(y, output_layer_output)
    d_output = error * sigmoid_derivada(output_layer_output)
    d_hidden = np.dot(d_output, peso2.T) * sigmoid_derivada(hidden_layer_output)
    peso2 +=  aprendizado_taxa * np.dot(hidden_layer_output.T, d_output)
    vies2 +=  aprendizado_taxa * np.sum(d_output, axis=0, keepdims=True)
    peso1 +=  aprendizado_taxa * np.dot(x.T, d_hidden)
    vies1 +=  aprendizado_taxa * np.sum(d_hidden, axis=0, keepdims=True)

    return peso1, vies1, peso2, vies2

## Função que ativa as funções anteriores, e contém o valor que determina a quantidade de repetições (epochs)
def fit(input:int, hidden:int, output:int, epochs:int, x:np.ndarray, y:np.ndarray, aprendizado_taxa:float):

    peso1, vies1, peso2, vies2 = inicializar_parametros(input, hidden, output)

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = propagacao_avancar(x, peso1, vies1, peso2, vies2)
        peso1, vies1, peso2, vies2 = propagacao_retroceder(x, y, hidden_layer_output, output_layer_output, peso1, vies1, peso2, vies2, aprendizado_taxa)

    return peso1, vies1, peso2, vies2

## Testa os dados através dos pesos obtidos
def prever(novo_input:np.ndarray, peso1:np.ndarray, vies1:np.ndarray, peso2:np.ndarray, vies2:np.ndarray):

    hidden_layer_input = np.dot(novo_input, peso1) + vies1
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, peso2) + vies2
    output_layer_output = sigmoid(output_layer_input)

    return output_layer_output


## Teste com valores reais
def main():

    input = 2
    hidden = 2
    output = 1
    aprendizado_taxa = 1.0 ## Controla a mudança do modelo em resposta a estimatimativa de erro a cada autalização dos pesos
    epochs = 1000 ## Repetições
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(1) ## Altera a semente para geração aleatória de números

    peso1, vies1, peso2, vies2 = fit(input, hidden, output, epochs, x, y, aprendizado_taxa)
    novo_input = np.array([[1, 1]])
    output_layer_output = prever(novo_input, peso1, vies1, peso2, vies2)

    print('Input  : ', novo_input)
    print('Output : ', np.round(output_layer_output))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
