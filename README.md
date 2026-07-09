# rag-finance — Asistente RAG sobre informes financieros (10-K)

Aplicación que responde preguntas en lenguaje natural sobre los informes anuales
(10-K) de Apple, Microsoft y NVIDIA usando RAG (Retrieval-Augmented Generation).

En lugar de que el LLM responda de memoria, el sistema primero recupera los
fragmentos más relevantes de los informes reales y se los pasa como contexto.
El resultado: respuestas fundamentadas en cifras concretas, con referencia al
documento fuente.

**Ejemplo:**

> *"What were Microsoft's main revenue sources in 2024?"*
>
> Based on the provided context, Microsoft's main revenue sources in 2024 were:
>
> - **Server products and cloud services:** $79,828 million
> - **Microsoft 365 Commercial:** $76,969 million
> - **Gaming:** $21,503 million
>
> *(Source: MSFT_10K.pdf, "Revenue, classified by significant product and service offerings")*

---

## Stack técnico

| Componente      | Tecnología                    |
| --------------- | ----------------------------- |
| Lenguaje        | Python 3.12+                  |
| Embeddings      | Gemini `gemini-embedding-001` |
| Generación      | Gemini `gemini-2.5-flash`     |
| Vector DB       | ChromaDB (local, persistente) |
| Interfaz        | Streamlit                     |
| Fuente de datos | SEC EDGAR (10-K públicos)     |

Sin frameworks de orquestación (LangChain, LlamaIndex) — pipeline construido
desde cero para entender cada paso.

---

## Arquitectura

```text
SEC EDGAR
    │
    ▼
download.py ──► data/raw/*.pdf
    │
    ▼
ingest.py ──► data/processed/chunks.jsonl
                    (chunks de 1000 tokens, overlap 200)
    │
    ▼
embed.py ──► ChromaDB (colección: finance_10k)
                    (gemini-embedding-001, RETRIEVAL_DOCUMENT)
    │
    ▼
         ┌─────────────────────────────┐
query ──►│ retrieve.py                 │
         │  embed query (RETRIEVAL_QUERY)│
         │  cosine similarity search   │
         │  filtro opcional por ticker │
         └──────────────┬──────────────┘
                        │ top-5 chunks
                        ▼
                  generate.py
                  (gemini-2.5-flash + contexto)
                        │
                        ▼
                    app.py (Streamlit)
```

---

## Estructura del repo

```text
.
├── data/
│   └── processed/
│       └── chunks.jsonl       # 1098 chunks indexados
├── src/
│   ├── download.py            # Descarga 10-Ks de SEC EDGAR
│   ├── ingest.py              # PDF → chunks JSONL
│   ├── embed.py               # Genera embeddings y los guarda en ChromaDB
│   ├── retrieve.py            # Búsqueda vectorial con filtro por empresa
│   ├── generate.py            # Llama a Gemini con el contexto recuperado
│   └── app.py                 # Interfaz Streamlit
├── .env                       # GEMINI_API_KEY (no versionado)
├── requirements.txt
└── .gitignore
```

---

## Cómo correrlo

### 1. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar API key

```bash
# Crear archivo .env en la raíz del proyecto:
GEMINI_API_KEY=tu_api_key_aqui
```

### 3. Descargar los 10-Ks

```bash
python src/download.py
```

### 4. Trocear los PDFs en chunks

```bash
python src/ingest.py
```

### 5. Generar embeddings e indexar en ChromaDB

```bash
python src/embed.py
# Nota: el free tier de Gemini permite 100 req/min y 1000 req/día.
# El script gestiona los rate limits automáticamente.
```

### 6. Lanzar la aplicación

```bash
streamlit run src/app.py
```

---

## Decisiones técnicas

### Chunk size: 1000 tokens, overlap 200

Los 10-K incluyen tablas financieras densas donde un número sin contexto no
tiene sentido. Con 1000 tokens el chunk suele contener una sección completa.
El overlap de 200 evita cortar una frase entre dos chunks consecutivos.

### ChromaDB local con distancia coseno

La similitud coseno mide el ángulo entre vectores, no su magnitud — lo que
importa para texto es la dirección semántica, no la longitud del fragmento.
ChromaDB local elimina latencia de red y coste de servicio en un prototipo.

### `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`

Gemini genera vectores distintos según el rol del texto. Los chunks se indexan
con `RETRIEVAL_DOCUMENT`; las preguntas se embeben con `RETRIEVAL_QUERY`.
Usar el mismo task_type para ambos degradaría la calidad de recuperación.

### Sin LangChain

Pipeline de ~6 funciones sin abstracciones intermedias. Cada paso es
inspeccionable directamente, lo que facilita depuración y entender qué ocurre
en cada fase del RAG.

---

## Roadmap

- [x] Descarga automática de 10-Ks desde SEC EDGAR
- [x] Chunking de PDFs (1000 tokens, overlap 200)
- [x] Embeddings + ChromaDB
- [x] Recuperación vectorial con filtro por empresa
- [x] Generación de respuestas con Gemini
- [x] Interfaz Streamlit
- [ ] Dataset de evaluación (20-30 preguntas con respuesta esperada)
- [ ] Métricas RAGAS (faithfulness, answer relevancy, context recall)
- [ ] Despliegue con URL pública
