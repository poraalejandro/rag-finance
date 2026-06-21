# rag-finance — Asistente RAG sobre informes financieros (10-K / 10-Q)

> Repo: https://github.com/poraalejandro/rag-finance
>
> Estado: 🚧 En construcción. Este README describe el plan; el código se irá
> completando y este documento se actualizará con resultados reales.

## Qué hace

Una aplicación que responde preguntas sobre informes anuales y trimestrales
(10-K / 10-Q) de un pequeño grupo de empresas cotizadas, usando RAG
(Retrieval-Augmented Generation): en vez de que el LLM responda solo de
memoria, primero busca los fragmentos de texto más relevantes de esos
informes en una base de datos vectorial y se los pasa como contexto antes
de generar la respuesta. Ejemplo de pregunta objetivo: *"¿Qué riesgos de
cadena de suministro menciona Apple en su último 10-K?"*

## Empresas cubiertas (versión inicial)

- [Empresa 1 — ej. Apple]
- [Empresa 2 — ej. Microsoft]
- [Empresa 3 — ej. NVIDIA]

Fuente de los documentos: [SEC EDGAR](https://www.sec.gov/edgar/search/) (10-K / 10-Q, gratuitos y públicos).

## Por qué este proyecto

Construido para demostrar manejo end-to-end de un sistema RAG en
producción: ingestión de documentos, chunking, embeddings, recuperación
híbrida, generación, y lo más importante — **evaluación medible**, no solo
una demo que "parece funcionar".

## Arquitectura

```
[Documentos] -> [Chunking] -> [Embeddings] -> [Vector DB]
                                                    |
[Pregunta usuario] -> [Embedding] -> [Retrieval] <-+
                                          |
                                   [Contexto + Pregunta]
                                          |
                                      [LLM] -> [Respuesta]
```

## Stack técnico

- **Lenguaje:** Python 3.11
- **Vector DB:** [pgvector / Chroma / Qdrant — elige uno, justifica por qué]
- **Embeddings:** [modelo elegido]
- **LLM:** [API elegida — Anthropic / OpenAI / Bedrock]
- **Framework de orquestación:** [LangChain / LlamaIndex / código propio]
- **Evaluación:** RAGAS
- **Despliegue:** Docker + [Render / Railway / Fly.io]

## Estructura del repo

```
.
├── data/
│   ├── raw/            # Documentos originales sin procesar
│   └── processed/       # Documentos ya troceados (chunks)
├── src/
│   ├── ingest.py        # Carga y trocea documentos
│   ├── embed.py         # Genera embeddings y los guarda en la vector DB
│   ├── retrieve.py      # Lógica de búsqueda/recuperación
│   ├── generate.py      # Llamada al LLM con el contexto recuperado
│   └── app.py            # API o interfaz de usuario
├── evals/
│   └── eval_dataset.json # Preguntas + respuestas esperadas para medir calidad
└── notebooks/             # Exploración y pruebas rápidas
```

## Cómo correrlo

```bash
# (a completar cuando el código esté listo)
pip install -r requirements.txt
python src/ingest.py
python src/app.py
```

## Resultados de evaluación

> Pendiente — aquí irán las métricas de RAGAS (faithfulness, answer
> relevancy, context precision/recall) una vez tengamos un primer pipeline
> funcionando, junto con una breve explicación de qué significan y cómo
> han mejorado iteración a iteración.

## Decisiones técnicas y por qué

> Pendiente — aquí explicarás, por ejemplo, por qué elegiste un tamaño de
> chunk concreto, por qué recuperación híbrida (BM25 + vectorial) en vez de
> solo vectorial, qué falló en la primera versión y cómo lo arreglaste.
> Esta sección es la que un entrevistador lee con más atención.

## Roadmap

- [ ] Ingestión y chunking de documentos
- [ ] Embeddings + vector DB funcionando
- [ ] Recuperación básica (solo vectorial)
- [ ] Generación de respuestas con contexto
- [ ] Dataset de evaluación (10-20 preguntas con respuesta esperada)
- [ ] Primera medición con RAGAS
- [ ] Recuperación híbrida (BM25 + vectorial) o reranking
- [ ] Segunda medición con RAGAS (comparar mejora)
- [ ] Dockerizar
- [ ] Desplegar con URL pública
- [ ] Interfaz simple (CLI o web mínima)
