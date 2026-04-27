"""
RAG-агент.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from retriever import HybridRetriever, SearchResult


class Chunk(BaseModel):
    """Один найденный фрагмент из векторного хранилища."""

    text: str
    chunk_id: int
    year: int
    distance: float


class RAGAgentAnswer(BaseModel):
    """Контракт ответа RAG-агента для системы оценки."""

    dataset_row_id: str | None = None
    answer: str
    retrieved_chunks: list[Chunk] | None = None


TOOLS_SCHEMA: list[dict[str, Any]] = []


def get_tool_functions(table: Any, client: Any) -> dict[str, Any]:
    retriever = HybridRetriever(
        table=table,
        client=client,
        embedding_model="text-embedding-3-small",
        rerank_model="openai/gpt-4o-mini",
    )
    return {"__retriever__": retriever}


class _AnswerPayload(BaseModel):
    answer: str = Field(..., description="Краткий точный ответ по контексту.")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ«\"])")


class RAGAgent:
    """Лёгкий RAG-пайплайн без внешнего оркестратора."""

    def __init__(
        self,
        client: Any,
        model: str,
        tools_schema: list[dict[str, Any]] | None = None,
        tool_functions: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools_schema = tools_schema or []
        self.tool_functions = tool_functions or {}

        retriever = self.tool_functions.get("__retriever__")
        if retriever is None:
            raise ValueError("В tool_functions не найден retriever")
        self.retriever: HybridRetriever = retriever

    def _build_context(self, results: list[SearchResult]) -> str:
        parts = []
        for item in results:
            parts.append(
                "\n".join(
                    [
                        f"[Год: {item.year}; Страница: {item.page}; Chunk ID: {item.chunk_id}]",
                        item.text.strip(),
                    ]
                )
            )
        return "\n\n---\n\n".join(parts)

    def _is_broad_question(self, question: str) -> bool:
        lowered = question.lower()
        markers = [
            "как измен",
            "каким образом",
            "какие факторы",
            "по сравнению",
            "на фоне",
            "соотно",
            "повли",
            "и как",
            "и какие",
        ]
        return any(marker in lowered for marker in markers)

    def _question_tokens(self, question: str) -> set[str]:
        text = question.lower().replace("ё", "е")
        tokens = set(re.findall(r"[\w%.-]+", text))
        stopwords = {
            "в",
            "во",
            "на",
            "по",
            "и",
            "как",
            "какой",
            "какая",
            "какие",
            "какое",
            "который",
            "которые",
            "что",
            "где",
            "когда",
            "сколько",
            "году",
            "год",
            "кв",
            "квартале",
            "квартал",
            "ii",
            "i",
            "это",
            "этот",
            "эта",
            "эти",
            "согласно",
            "данным",
        }
        return {
            t
            for t in tokens
            if t not in stopwords and not re.fullmatch(r"20\d{2}", t)
        }

    def _split_sentences(self, text: str) -> list[str]:
        prepared = re.sub(r"\s+", " ", text).strip()
        if not prepared:
            return []
        parts = _SENTENCE_SPLIT_RE.split(prepared)
        return [part.strip() for part in parts if part.strip()]

    def _sentence_score(self, sentence: str, question: str) -> float:
        qtokens = self._question_tokens(question)
        if not qtokens:
            return 0.0
        sentence_tokens = set(re.findall(r"[\w%.-]+", sentence.lower().replace("ё", "е")))
        overlap = len(qtokens & sentence_tokens)
        score = float(overlap)

        if re.search(r"\d", sentence):
            score += 0.35
        if any(token in sentence_tokens for token in ["ипу", "индекс", "джини", "росстат", "%"]):
            score += 0.1
        return score

    def _extractive_fallback(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "Не удалось найти подтверждённый ответ в документах."

        candidate_sentences: list[tuple[float, str]] = []
        for item in results[:2]:
            for sentence in self._split_sentences(item.text):
                candidate_sentences.append((self._sentence_score(sentence, question), sentence))

        candidate_sentences.sort(key=lambda x: x[0], reverse=True)
        candidate_sentences = [item for item in candidate_sentences if item[0] > 0]

        if not candidate_sentences:
            text = results[0].text.strip().replace("\n", " ")
            return text[:400]

        best = candidate_sentences[0][1]
        if not self._is_broad_question(question):
            return best

        merged = [best]
        for _, sentence in candidate_sentences[1:]:
            if sentence != best and len(merged) < 2:
                merged.append(sentence)
        return " ".join(merged)

    def _generate_answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "Не удалось найти подтверждённый ответ в документах."

        context = self._build_context(results)
        broad = self._is_broad_question(question)
        length_rule = (
            "Если вопрос широкий, дай максимум 2 коротких предложения."
            if broad
            else "Если вопрос factoid-style, дай ровно 1 короткое предложение или только значение/фразу."
        )

        system_prompt = (
            "Ты помогаешь отвечать на вопросы по отчётам НИУ ВШЭ.\n"
            "Отвечай только по предоставленному контексту.\n"
            "Нельзя придумывать факты, интерпретации и обобщения сверх текста.\n"
            "Нужно отвечать максимально extractive-style: брать формулировку как можно ближе к документу.\n"
            "Сохраняй точные числа, годы, проценты и названия показателей.\n"
            f"{length_rule}\n"
            "Не начинай ответ с фраз вроде 'Согласно контексту', 'В документе сказано', 'Исходя из данных'.\n"
            "Не используй фразы 'в контексте нет точного ответа', если в найденных фрагментах есть близкая прямая формулировка.\n"
            "Отказ допустим только если в переданном контексте действительно нет опоры для ответа.\n"
            "Верни JSON с единственным полем answer."
        )

        user_prompt = (
            f"Вопрос:\n{question}\n\n"
            f"Контекст:\n{context}\n\n"
            "Правила вывода:\n"
            "1. Сначала найди в контексте минимальный фрагмент, который отвечает на вопрос.\n"
            "2. Не пересказывай страницу целиком.\n"
            "3. Если вопрос просит число/дату/долю/место, верни именно это значение вместе с минимально нужной подписью.\n"
            "4. Если вопрос сравнительный или причинно-следственный, верни не более двух коротких предложений.\n"
            "5. Если ответа нет, верни: 'Не удалось найти подтверждённый ответ в документах.'"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            answer = str(data.get("answer", "")).strip()
            if answer:
                return answer
        except Exception:
            pass

        return self._extractive_fallback(question, results)

    def run(self, question: str, verbose: bool = False, dataset_row_id: str | None = None) -> RAGAgentAnswer:
        results = self.retriever.retrieve(question)
        answer = self._generate_answer(question, results)

        retrieved_chunks = [
            Chunk(
                text=item.text,
                chunk_id=item.chunk_id,
                year=item.year,
                distance=1.0 - float(item.score),
            )
            for item in results
        ]

        return RAGAgentAnswer(
            dataset_row_id=dataset_row_id,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )
