import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Verifica se o modelo está disponível
MODEL_PATH = "models/modelo_treinado.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("Modelo não encontrado. Execute o treinamento para gerar o modelo_treinado.pkl.")
    st.stop()

# Definição das features utilizadas no modelo
FEATURES = ['idade', 'renda', 'gasto_mensal', 'tempo_como_cliente']

# Layout - Tabs
tabs = st.tabs(["Cadastro de Cliente", "Avaliação de Cliente", "Analytics"])

# ========================== Tela 1: Cadastro de Cliente ==========================
with tabs[0]:
    st.header("Cadastro de Cliente")

    # Inputs
    cliente_nome = st.text_input("Nome do Cliente")
    cliente_data = {feature: st.number_input(f"Insira o valor para {feature}", value=0.0) for feature in FEATURES}
    cadastrar = st.button("Cadastrar Cliente")

    # Lista de Clientes Cadastrados (simulação local)
    if "clientes" not in st.session_state:
        st.session_state["clientes"] = []

    if cadastrar:
        cliente_data["nome"] = cliente_nome
        st.session_state["clientes"].append(cliente_data)
        st.success("Cliente cadastrado com sucesso!")

    st.write("Clientes Cadastrados:")
    st.write(pd.DataFrame(st.session_state["clientes"]))

# ========================== Tela 2: Avaliação de Cliente ==========================
with tabs[1]:
    st.header("Avaliação de Cliente")

    if len(st.session_state["clientes"]) == 0:
        st.warning("Nenhum cliente cadastrado.")
    else:
        cliente_idx = st.selectbox(
            "Selecione um cliente para avaliar:",
            range(len(st.session_state["clientes"])),
            format_func=lambda idx: st.session_state["clientes"][idx].get("nome", f"Cliente {idx + 1}")
        )
        cliente_selecionado = st.session_state["clientes"][cliente_idx]

        # Inicializa o threshold
        threshold = 0.5

        # Prompt para ajuste via texto
        prompt = st.text_input("Defina o threshold via texto (Ex: Defina o threshold para 0.75)", key="threshold_input_evaluation")
        try:
            if "threshold" in prompt.lower():
                threshold_value = float(prompt.split()[-1])
                if 0.0 <= threshold_value <= 1.0:
                    threshold = threshold_value
        except:
            st.warning("Formato do texto inválido. Use: 'Defina o threshold para X'")

        # Slider sincronizado com o threshold
        threshold = st.slider("Defina o threshold", 0.0, 1.0, threshold, 0.01)

        # Predição
        cliente_df = pd.DataFrame([cliente_selecionado]).drop(columns=["nome"])
        score = model.predict_proba(cliente_df)[0][1]
        resultado = 1 if score >= threshold else 0

        # Exibição da Predição
        st.subheader("Resultado da Predição")
        if resultado == 1:
            st.success(f"Propenso a comprar (Score: {score:.2%})")
        else:
            st.error(f"Não propenso a comprar (Score: {score:.2%})")

# ========================== Tela 3: Analytics ==========================
with tabs[2]:
    st.header("Analytics - Análise Comparativa por Classe")

    # Upload de CSV
    uploaded_file = st.file_uploader("Faça o upload do CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Verificação das colunas
        if not set(FEATURES).issubset(df.columns):
            st.error("O CSV deve conter as colunas: " + ", ".join(FEATURES))
        else:
            # Inicializa o threshold
            threshold = 0.5

            # Prompt para ajuste via texto
            prompt = st.text_input("Defina o threshold ou solicite análise (Ex: 'Defina o threshold para 0.75' ou 'Exibir os 5 maiores scores')", key="threshold_input_analytics")
            try:
                command = prompt.lower().strip()

                # Definição de threshold
                if "threshold" in command:
                    try:
                        threshold_value = float(command.split()[-1])
                        if 0.0 <= threshold_value <= 1.0:
                            threshold = threshold_value
                        else:
                            st.warning("O threshold deve estar entre 0.0 e 1.0.")
                    except ValueError:
                        st.warning("Formato esperado: 'Defina o threshold para X'.")
            except Exception as e:
                st.warning(f"Erro ao interpretar o comando: {str(e)}")

            # Slider sincronizado com o threshold
            threshold = st.slider("Defina o threshold para análise", 0.0, 1.0, threshold, 0.01)

            try:
                command = prompt.lower().strip()

                # Exibir maiores scores
                if "maiores scores" in command:
                    try:
                        n = int([word for word in command.split() if word.isdigit()][0])
                        top_scores = df.nlargest(n, "score")[["nome", "score"]]
                        st.write(f"Top {n} Maiores Scores:")
                        st.write(top_scores)
                    except (IndexError, ValueError):
                        st.warning("Formato esperado: 'Exibir os N maiores scores'.")

                # Exibir menores scores
                elif "menores scores" in command:
                    try:
                        n = int([word for word in command.split() if word.isdigit()][0])
                        bottom_scores = df.nsmallest(n, "score")[["nome", "score"]]
                        st.write(f"Top {n} Menores Scores:")
                        st.write(bottom_scores)
                    except (IndexError, ValueError):
                        st.warning("Formato esperado: 'Exibir os N menores scores'.")

                # Mostrar propensos a comprar
                elif "propensos a comprar" in command:
                    propensos = df[df["classe"] == 1][["nome", "score"]]
                    st.write("Clientes Propensos a Comprar:")
                    st.write(propensos)

                # Mostrar clientes com score acima de X
                elif "score acima de" in command:
                    try:
                        score_limit = float([word for word in command.split() if word.replace('.', '', 1).isdigit()][0])
                        filtered = df[df["score"] >= score_limit][["nome", "score"]]
                        st.write(f"Clientes com score acima de {score_limit}:")
                        st.write(filtered)
                    except (IndexError, ValueError):
                        st.warning("Formato esperado: 'Mostrar clientes com score acima de X'.")

            except Exception as e:
                st.warning(f"Erro ao interpretar o comando: {str(e)}")

            # Aplicar predição
            try:
                df["score"] = model.predict_proba(df[FEATURES])[:, 1]
                df["classe"] = (df["score"] >= threshold).astype(int)
            except Exception as e:
                st.warning(f"Erro ao aplicar predição: {str(e)}")

            st.write("Análise dos Dados")
            st.write(df.head())

            # Gráficos - Boxplot e Histogramas
            st.subheader("Distribuição das Features por Classe")

            cols = st.columns(2)
            for idx, feature in enumerate(FEATURES):
                with cols[idx % 2]:
                    st.markdown(f"**Distribuição de {feature}**")
                    # Boxplot
                    st.write("Boxplot")
                    fig, ax = plt.subplots()
                    df_plot = df[[feature, "classe"]].copy()
                    df_plot.boxplot(column=feature, by="classe", ax=ax)
                    ax.set_title(f"{feature} por classe")
                    ax.set_xlabel("Classe")
                    ax.set_ylabel(feature)
                    plt.suptitle("")
                    st.pyplot(fig)
                    plt.close(fig)

                    # Histograma
                    st.write("Histograma")
                    st.bar_chart(df.groupby("classe")[feature].mean())
