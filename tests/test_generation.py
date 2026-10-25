import json
from types import SimpleNamespace as NS

from rag_lab.chunking import Chunk
from rag_lab.generation import ABSTAIN, ClaudeGenerator, ExtractiveGenerator, sentences

IDF = {}
CTX = [Chunk("c1", "livrare", 0, 0, "Livrare › Costul livrării\nLivrarea standard costă 50 de lei. "
                                      "Pentru comenzile de peste 1000 de lei livrarea standard este gratuită."),
       Chunk("c2", "plati", 0, 0, "Plata ramburs este disponibilă pentru comenzile de până la 20 000 de lei.")]


def test_sentences_drop_headers_and_fragments():
    out = sentences("## Titlu\nDoc › Sec\nPrima propoziție aici. Da. A doua propoziție aici!")
    assert out == ["Prima propoziție aici.", "A doua propoziție aici!"]


def test_extractive_picks_best_sentence():
    ans = ExtractiveGenerator(IDF).answer("Cât costă livrarea standard?", CTX)
    assert not ans.abstained
    assert "50 de lei" in ans.text
    assert ans.input_tokens > 20


def test_extractive_abstains_when_nothing_matches():
    ans = ExtractiveGenerator(IDF).answer("Livrați cu drona în România?", CTX[1:])
    assert ans.abstained and ans.text == ABSTAIN


def test_threshold_controls_abstention():
    q = "Care e limita pentru ramburs?"
    assert not ExtractiveGenerator(IDF, threshold=0.2).answer(q, CTX).abstained
    assert ExtractiveGenerator(IDF, threshold=5.0).answer(q, CTX).abstained


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.params = payload, stop_reason, None
        self.messages = self

    def create(self, **params):
        self.params = params
        return NS(stop_reason=self.stop_reason, usage=NS(input_tokens=321, output_tokens=12),
                  content=[NS(type="text", text=json.dumps(self.payload))])


def test_claude_generator_uses_structured_output():
    client = FakeClient({"answer": "50 de lei", "abstain": False})
    ans = ClaudeGenerator(client=client).answer("Cât costă livrarea?", CTX)
    assert ans.text == "50 de lei" and ans.input_tokens == 321
    assert client.params["output_config"]["format"]["schema"]["required"] == ["answer", "abstain"]
    assert "[1]" in client.params["messages"][0]["content"]


def test_claude_generator_abstain_and_refusal():
    assert ClaudeGenerator(client=FakeClient({"answer": "", "abstain": True})).answer("?", CTX).abstained
    assert ClaudeGenerator(client=FakeClient({}, stop_reason="refusal")).answer("?", CTX).abstained
