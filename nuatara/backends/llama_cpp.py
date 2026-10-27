from llama_cpp import Llama
from nuatara.types import BackendInfo
import time

class LlamaCppBackend:
    name = "llama_cpp"

    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._model = None

    def load(self):
        self._model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=512,
            verbose=False,
        )

    def benchmark(self, n_tokens: int = 50) -> dict:
        prompt = "The quick brown fox"
        start = time.perf_counter()
        out = self._model(prompt, max_tokens=n_tokens, echo=False)
        elapsed = time.perf_counter() - start
        generated = out["usage"]["completion_tokens"]
        tokens_per_sec = generated / elapsed
        return {
            "tokens_per_sec": round(tokens_per_sec, 1),
            "tokens_generated": generated,
            "elapsed_sec": round(elapsed, 2),
        }

    def info(self) -> BackendInfo:
        import platform
        hw = "Apple Silicon" if platform.processor() == "arm" else "CPU"
        return BackendInfo(
            backend_name=self.name,
            hardware=hw,
            runtime="llama.cpp",
        )

    def release(self):
        self._model = None
