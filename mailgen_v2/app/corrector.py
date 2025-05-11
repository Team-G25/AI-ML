import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pykospacing import Spacing
from konlpy.tag import Okt

class KoreanTextCorrector:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
        self.tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.spacing = Spacing()
        self.okt = Okt()

    def correct_text(self, input_text):
        paragraphs = self._split_into_paragraphs(input_text)
        corrected_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                corrected_paragraphs.append("")
                continue

            spaced = self._clean_spacing(paragraph)
            sentences = self._split_into_sentences(spaced)
            corrected = self._correct_sentences_with_context(sentences)
            corrected_paragraphs.append(" ".join(corrected))

        full_text = "\n".join(corrected_paragraphs)
        cleaned_text = self._post_cleanup(full_text)
        final_text = self._fix_adverb_typos(cleaned_text)
        return final_text

    def _split_into_paragraphs(self, text):
        return text.split("\n")

    def _split_into_sentences(self, paragraph):
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        return [s.strip() for s in sentence_endings.split(paragraph) if s.strip()]

    def _extend_context(self, sentences, index, size=2):
        start = max(0, index - size)
        end = min(len(sentences), index + size + 1)
        return " ".join(sentences[start:end])

    def _clean_spacing(self, text):
        spaced = self.spacing(text)
        spaced = re.sub(r"\s+", " ", spaced)
        spaced = re.sub(r"\s+([.!?,])", r"\1", spaced)
        return spaced.strip()

    def _correct_sentences_with_context(self, sentences):
        corrected = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue
            context = self._extend_context(sentences, i)
            input_text = "맞춤법을 고쳐주세요: " + context
            encoding = self.tokenizer(input_text, return_tensors="pt")
            input_ids = encoding.input_ids.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            corrected.append(decoded)
        return corrected

    def _fix_adverb_typos(self, text):
        words = self.okt.pos(text, stem=True)
        corrected = []
        i = 0
        while i < len(words):
            word, tag = words[i]
            if i + 1 < len(words):
                next_word, next_tag = words[i + 1]
                if tag == "Adjective" and next_word == "개":
                    corrected.append(word + "게")
                    i += 2
                    continue
            corrected.append(word)
            i += 1
        return " ".join(corrected)

    def _post_cleanup(self, text):
        text = re.sub(r"(\d+)분 반", r"\1분반", text)
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r"(?<![a-zA-Z])매일(?=을 드리[고면었])", "메일", text)
        return text.strip()