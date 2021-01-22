from os import environ

import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
from psutil import cpu_count
from transformers import AutoTokenizer

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = "ACTIVE"


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

    assert (
        provider in get_all_providers()
    ), f"provider {provider} not found, {get_all_providers()}"

    # Few properties than might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1

    # Load the model as a graph and prepare the CPU backend
    return InferenceSession(model_path, options, providers=[provider])


class BertServer:
    def __init__(self) -> None:
        super().__init__()
        self.session = ort.InferenceSession("bert-base-cased.onnx")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def predict(self, text):
        onnx_in = self.tokenizer.encode_plus(text, return_tensors="np")
        inputs_onnx = {k: v for k, v in onnx_in.items()}
        _, pooled = self.session.run(None, inputs_onnx)
        return pooled
        # print(sequence.shape, pooled.shape)
