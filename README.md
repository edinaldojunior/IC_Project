## Contexto

O comportamento agressivo de condução é o principal fator de acidentes de trânsito. Conforme relatado pela Fundação AAA para Segurança no Trânsito , 106.727 acidentes fatais – 55,7% do total – durante um período recente de quatro anos envolveram motoristas que cometeram uma ou mais ações agressivas de direção. Portanto, como prever o comportamento de condução perigosa com rapidez e precisão?

## Abordagem da solução

A condução agressiva inclui excesso de velocidade, frenagens repentinas e curvas repentinas à esquerda ou à direita. Todos esses eventos são refletidos nos dados do acelerômetro e do giroscópio. Por isso, sabendo que hoje em dia quase todo mundo possui um smartphone que possui uma grande variedade de sensores, foi utilizado dados de um aplicativo de coleta de dados em Android baseado nos sensores acelerômetro e giroscópio, para realizar a classificação do comportamento de motoristas, a partir do uso dos classificadores:
    
   * CatBoost
   * LightGBM
   * XGBoost
   * Ensemble com os classificadores 

Para este projeto, foi utilizada uma técnica de redução de dimensionalidade, que visa escolher um subconjunto de recursos relevantes dos recursos originais, removendo recursos irrelevantes, redundantes ou ruidosos. A seleção de recursos geralmente pode levar a um melhor desempenho de aprendizado, maior precisão de aprendizado, menor custo computacional e melhor interpretabilidade do modelo. Este Trabalho utilizou o Binary Fish School Search Algorithm no processo de seleção de recursos, aumentando a precisão do classificador e reduzindo o número de atributos para a tarefa de classificação.

O ponto crítico para encontrar os melhores modelos que podem resolver um problema não são apenas os tipos de modelos. É preciso encontrar os parâmetros ideais para que os modelos funcionem de maneira ideal, dado o conjunto de dados. Isso é chamado de localizar ou pesquisar hiperparâmetros. Neste trabalho utilizamos os seguintes algoritmos para este objetivo:
    

   * Particle Swarm Optimization (PSO)
   * Genetic Algorithm

A acurácia, AUC, Precisão, Recall são considerados através da realização de uma avaliação de aptidão. Os algoritmos de otimização foram avaliados usando os classificadores citados acima. O conjunto de dados Driving Behavior foi usado para treinar e avaliar os algoritmos. Os resultados mostram que o método é útil para reduzir o tempo de treinamento e aumentar a assertividade.

Driving Behavior: https://www.kaggle.com/datasets/outofskills/driving-behavior

## Dataset:
* Taxa de amostragem: 2 amostras (linhas) por segundo.
* Aceleração gravitacional: removida.
* Sensores: Acelerômetro e Giroscópio.

### Dados:

* Aceleração (eixo X,Y,Z em metros por segundo ao quadrado (m/s2))
* Rotação (eixo X,Y, Z em graus por segundo (°/s))
* Etiqueta de classificação (LENTO, NORMAL, AGRESSIVO)
* Timestamp (tempo em segundos)

Comportamentos de condução:

* Lento
* Normal
* Agressivo


## Resultados
Durante a execução, as simulações mostraram que o método baseado em BFSS e a otimização dos hiperparâmetros com GA, foi capaz de melhorar o desempenho dos métodos de aprendizado de máquina, principalmente o Ensemble. As simulações também demonstraram que o método proposto supera a utilização de outros classificadores, utilizando PSO. Além disso, o uso do BFSS para seleção de atributos reduziu consideravelmente o número de atributos de entrada usados ​​em nossa análise.

## Conclusão
A combinação de seleção de recursos de entrada e otimização de parâmetros de métodos de aprendizado de máquina melhora a precisão do esforço de desenvolvimento de software. Além disso, isso reduz a complexidade do modelo, o que pode ajudar a entender a relevância de cada recurso de entrada. Portanto, alguns parâmetros de entrada podem ser ignorados sem perda de precisão nas estimativas. 

Foi mostrado que é possível explorar um vasto espaço de parâmetros-modelo possíveis de forma eficiente, economizando tempo e recursos computacionais. Isso é realizado formulando a busca por hiperparâmetros como uma função objetiva para a qual encontramos os valores óptimos usando algoritmos de otimização.
