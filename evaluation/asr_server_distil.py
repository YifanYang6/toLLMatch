import os, sys, argparse, copy
from termcolor import cprint
from dotenv import load_dotenv
from typing import List

import msgpack
from termcolor import cprint

load_dotenv("../.env", override=True)  # load API keys into
print(os.getenv("HF_HOME"))
sys.path.append("../")

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, Response


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--srate", type=int, default=16000)
parser.add_argument("--device", type=int, default=0)
args, unknown_args = parser.parse_known_args()
cprint(f"args: {args.model_path}", "green")
MIN_LAG_WORDS = 2
DEVICE = f"cuda:{args.device}"
SRATE = args.srate
ASR_MODEL_NAME=args.model_path
DEVICE = f"cuda:{args.device}"
SRATE = args.srate


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()


class AsrWhisperDistil:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)

        self.srate = SRATE
        self.DEVICE = DEVICE

        cprint(f"using: {ASR_MODEL_NAME}", "green", attrs=["bold"])
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(DEVICE)
        self.p_generate = self.generate_fn

    def generate_fn(self, input_features):
        cprint(f"input_features: {input_features.shape}", color="yellow")
        pred_ids = self.model.generate(
            input_features,
            return_timestamps=False,
            max_new_tokens=128,
            num_beams=1,
        )
        return pred_ids

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:
        input_features = self.processor(
            audio_array,
            sampling_rate=self.srate,
            return_tensors="pt",
        ).input_features
        predicted_ids = self.p_generate(input_features.to(DEVICE, dtype=torch.float16))
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        cprint(f"audio_array len: {len(audio_array)} | transcription: {transcription}", color="yellow")
        # NOTE!!! postprocess on the client side
        # return self._postprocess(transcription, source_finished)
        return transcription


asr_model = AsrWhisperDistil(ASR_MODEL_NAME, DEVICE, SRATE)


async def unpack_msgpack(request: Request):
    """Unpack MessagePack request body. This is because we're recieving MessagePack data (for speed)"""
    body_bytes = await request.body()  # Read the full body as bytes
    data = msgpack.unpackb(body_bytes)  # Unpack the bytes using MessagePack
    return data


@app.post("/info")
async def get_info(request: Request) -> Response:
    return JSONResponse({"asr_model_name": ASR_MODEL_NAME})


@app.post("/generate")
async def recognize(request_dict: dict = Depends(unpack_msgpack)) -> Response:
    # request_dict = await request.json()
    source = copy.deepcopy(request_dict["source"])  # FIXME: return to .pop() after debugging
    source_finished = request_dict.pop("source_finished")
    recognized_word_list = asr_model.recognize(source, source_finished)
    # cprint(f"recognized_word_list: {recognized_word_list}", color="yellow")
    return JSONResponse(
        {"recognized_word_list": recognized_word_list}
    )  # NOTE: list of recognized texts (becuase a batch might have multiple audio files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--root-path", type=str, default=None, help="FastAPI root_path when app is behind a path based routing proxy"
    )

    args, _ = parser.parse_known_args()
    app.root_path = args.root_path
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
