# lang-chain-python
learning project for AI process chaining and RAG

## ⚙️ Guia de Configuração

Siga os passos abaixo para configurar seu ambiente e utilizar os scripts do projeto.

### 1. Criar e Ativar Ambiente Virtual

**Windows:**
```bash
python -m venv langchain
langchain\Scripts\activate
```

**Mac/Linux:**
```bash
pyenv install 3.10.20
pyenv virtualenv 3.10.20 langchain
source /home/USER/.pyenv/versions/langchain/bin/activate
```

### 2. Instalar Dependências

Utilize o comando abaixo para instalar as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 3. Configurar Chave da OpenAI

Crie ou edite o arquivo `.env` adicionando sua chave de API da OpenAI:
```bash
OPENAI_API_KEY="SUA_CHAVE_DE_API"
```

### Models

Use either `gpt-4o-mini` or `gpt-5-nano` for cheapest prices (as of March 2026)