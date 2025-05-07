# NomeDoProjeto

Sistema para predição de propensão de compra utilizando Machine Learning em Streamlit.

Estrutura do projeto:
- Cadastro de Cliente
- Avaliação de Cliente
- Analytics com gráficos dinâmicos

# Sistema de Predição de Propensão de Compra

Aplicativo desenvolvido em Streamlit para prever a propensão de compra de clientes utilizando Machine Learning.

## Funcionalidades do Sistema

1. **Cadastro de Cliente:**
   - Insira os dados do cliente (`Idade`, `Renda`, `Gasto Mensal`, `Tempo como Cliente`).
   - Os clientes cadastrados são armazenados e listados para avaliação posterior.

2. **Avaliação de Cliente:**
   - Selecione um cliente cadastrado para análise.
   - Defina o threshold utilizando um slider ou via comando de texto. Exemplo: "Defina o threshold para 0.75".
   - Realize a predição com base no modelo treinado e exiba o resultado em destaque:
     - ✅ Verde: Propenso a comprar (`1`).
     - ❌ Vermelho: Não propenso (`0`).

3. **Analytics:**
   - Realize o upload de um CSV contendo dados de clientes.
   - Ajuste o threshold e visualize gráficos dinâmicos (boxplot/histograma) comparando as features entre `0` e `1`.
   - Os gráficos se atualizam automaticamente ao alterar o threshold.

## Estrutura do Projeto

```
MeuProjeto/
├─ app.py                    # Aplicativo principal em Streamlit
├─ train_model.py            # Script para treinar o modelo
├─ requirements.txt          # Dependências do projeto
├─ README.md                 # Documentação do projeto
├─ models/
│   └─ modelo_treinado.pkl   # Modelo treinado em formato .pkl
├─ data/
│   └─ exemplo_batch.csv     # Dataset de exemplo para o módulo Analytics
├─ utils/
│   └─ __init__.py
│   └─ predicao.py           # Funções de predição
│   └─ plots.py              # Funções para geração de gráficos
├─ docs/                     # Documentação adicional
└─ tests/                    # Testes unitários
```

## Instalação e Execução

1. Crie um ambiente virtual:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Execute o script de configuração:
   ```
   ./setup_project.sh
   ```

3. Inicie o aplicativo:
   ```
   streamlit run app.py
   ```

4. Para treinar o modelo novamente:
   ```
   python train_model.py
   ```