from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from typing import Iterable

from .contracts import CandidateSegmentation, RawDocument


@dataclass(frozen=True)
class AlignmentFailure:
    doc_id: str
    system: str
    reason: str
    offset: int | None = None
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "system": self.system,
            "reason": self.reason,
            "offset": self.offset,
            "context": self.context,
        }


def version_of(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def segment_offsets(text: str, segments: Iterable[str]) -> tuple[int, ...]:
    """Map text-preserving segments to end offsets without fuzzy matching."""
    pieces = list(segments)
    if not pieces:
        raise ValueError("Splitter returned no segments")
    reconstructed = "".join(pieces)
    if reconstructed != text:
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(reconstructed, text)) if left != right),
            min(len(reconstructed), len(text)),
        )
        raise ValueError(
            f"Splitter output does not reconstruct input exactly at char {mismatch}; "
            f"output_length={len(reconstructed)} input_length={len(text)}"
        )
    offsets: list[int] = []
    cursor = 0
    for piece in pieces[:-1]:
        cursor += len(piece)
        boundary = cursor
        while boundary > 0 and text[boundary - 1].isspace():
            boundary -= 1
        offsets.append(boundary)
    return tuple(sorted(set(offsets)))


def canonicalize_offsets(
    document: RawDocument,
    raw_offsets: Iterable[int],
    canonical_offsets: set[int],
    system: str,
) -> tuple[tuple[int, ...], list[AlignmentFailure]]:
    valid: list[int] = []
    failures: list[AlignmentFailure] = []
    for offset in sorted(set(raw_offsets)):
        if offset == len(document.text):
            continue
        if offset in canonical_offsets:
            valid.append(offset)
            continue
        start, end = max(0, offset - 35), min(len(document.text), offset + 35)
        failures.append(AlignmentFailure(
            document.doc_id,
            system,
            "predicted boundary is not a canonical inter-token whitespace gap",
            offset,
            document.text[start:end],
        ))
    return tuple(valid), failures


class RealSplitterSuite:
    """Lazy adapters for Punkt, Stanza, WtP and SaT on identical raw text."""

    def __init__(self, *, stanza_dir: str, use_gpu: bool = True):
        self.stanza_dir = stanza_dir
        self.use_gpu = use_gpu
        self._punkt = None
        self._stanza = None
        self._wtp = None
        self._sat = None

    def _load_punkt(self):
        if self._punkt is None:
            import nltk

            nltk_data = os.environ.get("NLTK_DATA")
            if nltk_data and nltk_data not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data)
            self._punkt = nltk.data.load("tokenizers/punkt/english.pickle")
        return self._punkt

    def _load_stanza(self):
        if self._stanza is None:
            import stanza

            self._stanza = stanza.Pipeline(
                lang="en",
                processors="tokenize",
                dir=self.stanza_dir,
                use_gpu=self.use_gpu,
                download_method=None,
                verbose=False,
            )
        return self._stanza

    def _load_wtp(self):
        if self._wtp is None:
            from wtpsplit import WtP

            self._wtp = WtP("wtp-bert-mini")
            if self.use_gpu:
                self._wtp.half().to("cuda")
        return self._wtp

    def _load_sat(self):
        if self._sat is None:
            from wtpsplit import SaT

            # General small-model checkpoint: deliberately no UD/GUM LoRA.
            self._sat = SaT("sat-3l-sm")
            if self.use_gpu:
                self._sat.half().to("cuda")
        return self._sat

    def predict_raw_offsets(self, system: str, texts: list[str]) -> list[tuple[int, ...]]:
        if system == "punkt":
            tokenizer = self._load_punkt()
            return [tuple(end for _, end in tokenizer.span_tokenize(text)) for text in texts]
        if system == "stanza":
            pipeline = self._load_stanza()
            outputs: list[tuple[int, ...]] = []
            for text in texts:
                parsed = pipeline(text)
                outputs.append(tuple(
                    max(token.end_char for token in sentence.tokens)
                    for sentence in parsed.sentences if sentence.tokens
                ))
            return outputs
        if system in {"wtp", "sat"}:
            model = self._load_wtp() if system == "wtp" else self._load_sat()
            split_outputs = list(model.split(texts))
            if len(split_outputs) != len(texts):
                raise RuntimeError(
                    f"{system} returned {len(split_outputs)} documents for {len(texts)} inputs"
                )
            return [segment_offsets(text, segments) for text, segments in zip(texts, split_outputs)]
        raise ValueError(f"Unknown splitter: {system}")

    @staticmethod
    def system_metadata(system: str) -> tuple[str, str]:
        if system == "punkt":
            return version_of("nltk"), "nltk english.pickle; default parameters"
        if system == "stanza":
            return version_of("stanza"), "en tokenize processor; raw text; default model"
        if system == "wtp":
            return version_of("wtpsplit"), "wtp-bert-mini; default threshold"
        if system == "sat":
            return version_of("wtpsplit"), "sat-3l-sm; default threshold; no LoRA"
        raise ValueError(f"Unknown splitter: {system}")

    def predict(
        self,
        system: str,
        documents: list[RawDocument],
        canonical_by_doc: dict[str, set[int]],
    ) -> tuple[list[CandidateSegmentation], list[AlignmentFailure]]:
        raw_by_doc = self.predict_raw_offsets(system, [doc.text for doc in documents])
        version, configuration = self.system_metadata(system)
        predictions: list[CandidateSegmentation] = []
        failures: list[AlignmentFailure] = []
        for document, raw_offsets in zip(documents, raw_by_doc):
            offsets, doc_failures = canonicalize_offsets(
                document, raw_offsets, canonical_by_doc[document.doc_id], system
            )
            failures.extend(doc_failures)
            predictions.append(CandidateSegmentation(
                doc_id=document.doc_id,
                text_sha256=document.sha256,
                system=system,
                offsets=offsets,
                system_version=version,
                configuration=configuration,
            ))
        return predictions, failures
