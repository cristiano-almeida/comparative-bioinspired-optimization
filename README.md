---

## 🌀 Otimização de Mola de Tensão/Compressão

### Uma análise comparativa entre Estratégias Evolutivas e Meta-heurísticas Bioinspiradas

Este repositório apresenta a **demonstração funcional** do artigo:
**"Otimização de Mola de Tensão/Compressão: uma análise comparativa entre estratégias evolutivas e meta-heurísticas bioinspiradas"**.

---

### 📄 Resumo do Artigo

Este estudo compara a aplicação de **Estratégias Evolutivas (ES)** com o algoritmo **Harris Hawks Optimization (HHO)** na otimização de uma mola de tensão/compressão com restrições. Foram testadas cinco estratégias de penalização (fixa, progressiva, híbrida, violação máxima e sem penalização) usando a biblioteca **MEALPY**.

📌 **Melhor desempenho com ES:**
Peso mínimo = **0,0127338 kg** (violação máxima)
📌 **Melhor resultado geral (literatura):**
HHO = **0,012665443 kg**

O estudo mostra como diferentes penalizações impactam na viabilidade e convergência das soluções.

---

## ⚙️ Como Executar o Código (passo a passo)

### ✅ Pré-requisito: Ter o Python instalado

Baixe e instale o Python 3.8 ou superior:
🔗 [https://www.python.org/downloads/](https://www.python.org/downloads/)

> Verifique se o Python está instalado corretamente:
> Abra o terminal (ou Prompt de Comando) e digite:

```bash
python --version
```

---

### ⬇️ 1. Baixe este repositório

* Clique em **"Code"** (canto superior direito da página)
* Depois em **"Download ZIP"**
* Extraia o conteúdo do ZIP em uma pasta local

---

### 🧪 2. Crie e ative um ambiente virtual (recomendado)

No terminal, navegue até a pasta extraída e digite:

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 📦 3. Instale as dependências

Ainda dentro do ambiente virtual, execute:

```bash
pip install -r requirements.txt
```

---

### ▶️ 4. Execute o código principal

```bash
python main.py
```

Isso irá executar os testes de otimização para diferentes estratégias de penalização.

---

## 📁 Estrutura do Projeto

```bash
.
├── main.py              # Código principal da análise
├── requirements.txt     # Lista de dependências (inclui MEALPY)
├── results/             # Pasta para receber resultados gerados durante a execução
├── my_results/          # Resultados oficiais do artigo (não precisa reexecutar)
└── README.md            # Este guia de execução
```

---

## 📊 Sobre os Resultados

A pasta `my_results/` contém os **resultados oficiais do artigo**, incluindo os valores ótimos encontrados por cada estratégia.
Esses arquivos permitem análise direta **sem necessidade de reexecutar o código**.

---

## 🔬 Tecnologias e Algoritmos Usados

* **Python 3.8+**
* **MEALPY** – Framework para meta-heurísticas
* **HHO (Harris Hawks Optimization)**
* **ES (Evolutive Strategy)** com 5 penalizações:

  * Fixa
  * Progressiva
  * Híbrida
  * Violação máxima
  * Sem penalização

---

## ✍️ Créditos

Este repositório acompanha o artigo científico como material complementar para fins **educacionais e demonstrativos**, voltado a estudantes e pesquisadores de Otimização e Inteligência Computacional.

---
