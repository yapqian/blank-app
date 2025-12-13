import re
import torch

def stub_load_translator_model():
    class Tok:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {'input_ids': torch.tensor([[1,2,3]])}
        def encode(self, text, add_special_tokens=False):
            return [1,2,3]
        def decode(self, ids, skip_special_tokens=True):
            return "Halo dunia"
    class Model:
        def generate(self, **kwargs):
            return [[1,2,3]]
    return Tok(), Model(), torch.device('cpu')


def translate_en_to_malay(text: str, max_length: int = 512) -> str:
    text = text.strip()
    if text == "":
        return ""
    translator_tok, translator_model, device = stub_load_translator_model()
    prefix = "terjemah ke Melayu: "
    input_text = prefix + text

    def _chunk_text_for_translation(s: str, tokenizer, max_len: int):
        s = s.strip()
        if not s:
            return []
        sents = re.split(r'(?<=[.!?])\s+', s)
        chunks = []
        cur = ""
        for sent in sents:
            cand = (cur + " " + sent).strip() if cur else sent
            try:
                tok_len = len(tokenizer.encode(cand, add_special_tokens=False))
            except Exception:
                tok_len = len(cand.split())
            if tok_len <= max_len - 16:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                try:
                    sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
                except Exception:
                    sent_len = len(sent.split())
                if sent_len > max_len - 16:
                    chunks.append(sent[: max_len * 2])
                    cur = ""
                else:
                    cur = sent
        if cur:
            chunks.append(cur)
        return chunks

    chunks = _chunk_text_for_translation(input_text, translator_tok, max_length)
    if not chunks:
        return ""

    translated_parts = []
    for chunk in chunks:
        inputs = translator_tok(chunk, return_tensors="pt", truncation=True, max_length=max_length)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = translator_model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
        out = translator_tok.decode(outputs[0], skip_special_tokens=True)
        translated_parts.append(out)

    translated = " ".join(translated_parts)
    print(translated)


if __name__ == '__main__':
    translate_en_to_malay('Hello world')
