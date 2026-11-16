from __future__ import annotations

import time


class RAGPipeline:
    def __init__(
        self,
        asr,
        encoder,
        retriever,
        data,
        context_builder,
        generator,
        reranker=None,
        candidate_depth: int | None = None,
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator
        # Optional reranking stage (investigation): retrieve candidate_depth
        # records densely, rerank them with a cross-encoder, keep the top k.
        self.reranker = reranker
        self.candidate_depth = candidate_depth
        if reranker is not None and (candidate_depth is None or candidate_depth < 1):
            raise ValueError("candidate_depth is required when a reranker is used")

    def answer_audio(self, audio_path, k: int = 5):
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio_path)
        asr_ms = (time.perf_counter() - t0) * 1000.0

        result = self.answer_transcript(transcript, k=k)
        result["timing"] = {"asr_ms": asr_ms, **result["timing"]}
        result["timing"]["end_to_end_ms"] = (time.perf_counter() - total_start) * 1000.0
        return result

    def answer_transcript(self, transcript: str, k: int = 5):
        """Everything after ASR: embed, retrieve (and rerank), build context, generate."""
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        rerank = None
        if self.reranker is None:
            retrieval = self.retriever.retrieve(query_vector, k=k)
            record_ids = retrieval["record_ids"]
        else:
            if self.candidate_depth < k:
                raise ValueError("candidate_depth must be at least k")
            retrieval = self.retriever.retrieve(query_vector, k=self.candidate_depth)
            rerank = self.reranker.rerank(transcript, retrieval["record_ids"], self.data, k)
            record_ids = rerank["record_ids"]

        built = self.context_builder.build(
            transcript,
            record_ids,
            self.data,
        )

        generation = self.generator.generate(built["prompt"])

        total_ms = (time.perf_counter() - total_start) * 1000.0

        timing = {
            "embedding_ms": embedding_ms,
            "retrieval_ms": retrieval["latency_ms"],
        }
        if rerank is not None:
            timing["rerank_ms"] = rerank["latency_ms"]
        timing["generation_ms"] = generation["latency_ms"]
        timing["end_to_end_ms"] = total_ms

        return {
            "transcript": transcript,
            "retrieval": retrieval,
            "rerank": rerank,
            "record_ids": record_ids,
            "contexts": built["contexts"],
            "prompt_chars": len(built["prompt"]),
            "answer": generation["answer"],
            "timing": timing,
        }
