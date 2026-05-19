import json
import time
import urllib.error
import urllib.parse
import urllib.request

from openai import OpenAI


class ChatSession:
    def __init__(
        self,
        ip,
        port,
        context_separator="###",
        use_batching=False,
        batch_size=1,
        scheduler="none",
        batch_timeout_ms=50,
        use_rag=False,
        rag_request_id=None,
        rag_document_set="base",
    ):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://{ip}:{port}/v2"
        self.openai_api_base = openai_api_base

        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id

        self.messages = []

        self.final_context = ""
        self.context_separator = context_separator
        self.extra_headers = {}
        if use_batching:
            self.extra_headers.update({
                "x-lmcache-batch-size": str(max(1, batch_size)),
                "x-lmcache-scheduler": scheduler,
                "x-lmcache-batch-timeout-ms": str(max(1, batch_timeout_ms)),
            })
        if use_rag:
            self.extra_headers.update({
                "x-lmcache-rag": "true",
                "x-lmcache-rag-docset": rag_document_set,
            })
            if rag_request_id is not None:
                self.extra_headers["x-lmcache-rag-request-id"] = str(rag_request_id)
        self.rag_request_id = rag_request_id


    def set_context(self, context_strings):
        contexts = []
        for context in context_strings:
            contexts.append(context)

        self.final_context =  self.context_separator.join(contexts) 
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it!", display=False)

    def get_context(self):
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print("User message:", message)
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print("Server message:", message)
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, question):
        self.on_user_message(question)

        start = time.perf_counter()
        end = None
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=0.5,
            stream=True,
            stop = "\n",
            extra_headers=self.extra_headers or None,
        )

        server_message = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if end is None:
                    end = time.perf_counter()
                yield chunk_message
                server_message.append(chunk_message)

        self.on_server_message("".join(server_message))
        yield f"\n\n(Response delay: {end - start:.2f} seconds)"

    def get_rag_metrics(self):
        if self.rag_request_id is None:
            return None

        escaped_request_id = urllib.parse.quote(str(self.rag_request_id), safe="")
        url = f"{self.openai_api_base}/rag/metrics/{escaped_request_id}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
