"""S2SEmoControl dataset generation pipeline."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ark import ArkWriter
from common.jsonl import write_jsonl_record
from .audio_ms import process_tts_output, save_wav
from .config import BATCH_SIZE, DEFAULT_MAX_WORKERS, TTS_SAMPLE_RATE, WAV_SAMPLE_RATE
from .cosyvoice_npu import CosyVoiceNPUBackend
from .dataset_schema import build_synthesis_inputs, prepare_line_dict


@dataclass(frozen=True)
class S2SOutputPaths:
    query_token_ark: Path
    query_token_scp: Path
    query_audio_ark: Path
    query_audio_scp: Path
    answer_token_ark: Path
    answer_token_scp: Path
    answer_audio_ark: Path
    answer_audio_scp: Path
    jsonl: Path
    query_wav_dir: Path
    answer_wav_dir: Path

    @classmethod
    def build(cls, meta_path: str, output_path: str, shard_id: int) -> "S2SOutputPaths":
        meta_name = Path(meta_path).stem
        out = Path(output_path)
        return cls(
            query_token_ark=out / f"{meta_name}_query_token_{shard_id}.ark",
            query_token_scp=out / f"{meta_name}_query_token_{shard_id}.scp",
            query_audio_ark=out / f"{meta_name}_query_audio_{shard_id}.ark",
            query_audio_scp=out / f"{meta_name}_query_audio_{shard_id}.scp",
            answer_token_ark=out / f"{meta_name}_answer_token_{shard_id}.ark",
            answer_token_scp=out / f"{meta_name}_answer_token_{shard_id}.scp",
            answer_audio_ark=out / f"{meta_name}_answer_audio_{shard_id}.ark",
            answer_audio_scp=out / f"{meta_name}_answer_audio_{shard_id}.scp",
            jsonl=out / f"{meta_name}_{shard_id}.jsonl",
            query_wav_dir=out / f"{meta_name}_{shard_id}_query_wav",
            answer_wav_dir=out / f"{meta_name}_{shard_id}_answer_wav",
        )

    def prepare(self) -> None:
        self.query_wav_dir.mkdir(parents=True, exist_ok=True)
        self.answer_wav_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)


class S2SEmoControlPipeline:
    def __init__(self, backend: CosyVoiceNPUBackend, audio_dict: Dict[str, Dict[str, Any]]):
        self.backend = backend
        self.audio_dict = audio_dict

    def _worker_process(self, sample: Tuple[int, Dict[str, Any]]):
        sample_id, line_dict = sample
        new_line_dict = prepare_line_dict(line_dict)
        language = new_line_dict["language"]

        new_line_dict, q_instruct, q_content, a_instruct, a_content = build_synthesis_inputs(
            new_line_dict
        )

        text_prompt_key = new_line_dict["query_id"]
        answer_prompt_key = new_line_dict["answer_id"]

        q_tokens, q_speech = self.backend.compute_zeroshot_speech_token(
            q_instruct,
            self.audio_dict[language][text_prompt_key],
            q_content,
        )
        a_tokens, a_speech = self.backend.compute_zeroshot_speech_token(
            a_instruct,
            self.audio_dict[language][answer_prompt_key],
            a_content,
        )

        q_ark_pcm, q_wav_ms = process_tts_output(q_speech, TTS_SAMPLE_RATE)
        a_ark_pcm, a_wav_ms = process_tts_output(a_speech, TTS_SAMPLE_RATE)

        return (
            sample_id,
            new_line_dict,
            q_tokens,
            q_ark_pcm,
            q_wav_ms,
            a_tokens,
            a_ark_pcm,
            a_wav_ms,
        )

    def run(
        self,
        args,
        id2meta: List[Tuple[int, Dict[str, Any]]],
        parallel_idx: int,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        paths = S2SOutputPaths.build(args.meta_path, args.output_path, parallel_idx)
        paths.prepare()

        print(
            "writing to:\n"
            f"  {paths.query_token_ark}\n"
            f"  {paths.query_audio_ark}\n"
            f"  {paths.answer_token_ark}\n"
            f"  {paths.answer_audio_ark}\n"
            f"  {paths.jsonl}"
        )

        with (
            ArkWriter(paths.query_token_ark, paths.query_token_scp) as query_token_writer,
            ArkWriter(paths.query_audio_ark, paths.query_audio_scp) as query_audio_writer,
            ArkWriter(paths.answer_token_ark, paths.answer_token_scp) as answer_token_writer,
            ArkWriter(paths.answer_audio_ark, paths.answer_audio_scp) as answer_audio_writer,
            paths.jsonl.open("a+", encoding="utf-8") as jsonl_f,
        ):
            print(
                "resume offsets: "
                f"query_token={query_token_writer.offset}, "
                f"query_audio={query_audio_writer.offset}, "
                f"answer_token={answer_token_writer.offset}, "
                f"answer_audio={answer_audio_writer.offset}"
            )

            total = len(id2meta)
            for batch_start in range(0, total, BATCH_SIZE):
                batch = id2meta[batch_start: batch_start + BATCH_SIZE]
                results = []

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._worker_process, sample): sample[0]
                        for sample in batch
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"batch {batch_start // BATCH_SIZE + 1}",
                    ):
                        sid = futures[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            print(f"Error processing sample {sid}: {exc}")
                            print(traceback.format_exc())

                results.sort(key=lambda x: x[0])
                for res in results:
                    (
                        sample_id,
                        record,
                        q_tokens,
                        q_ark_pcm,
                        q_wav_ms,
                        a_tokens,
                        a_ark_pcm,
                        a_wav_ms,
                    ) = res

                    query_token_offset = query_token_writer.write(sample_id, q_tokens.tobytes())
                    query_audio_offset = query_audio_writer.write(sample_id, q_ark_pcm.tobytes())
                    answer_token_offset = answer_token_writer.write(sample_id, a_tokens.tobytes())
                    answer_audio_offset = answer_audio_writer.write(sample_id, a_ark_pcm.tobytes())

                    record["query_token_25hz"] = f"{paths.query_token_ark}:{query_token_offset}"
                    record["query_audio_ark"] = f"{paths.query_audio_ark}:{query_audio_offset}"
                    record["query_audio_path"] = str(
                        paths.query_wav_dir / f"{record['uuid']}-query.wav"
                    )
                    record["answer_token_25hz"] = f"{paths.answer_token_ark}:{answer_token_offset}"
                    record["answer_audio_ark"] = f"{paths.answer_audio_ark}:{answer_audio_offset}"
                    record["answer_audio_path"] = str(
                        paths.answer_wav_dir / f"{record['uuid']}-answer.wav"
                    )

                    save_wav(record["query_audio_path"], q_wav_ms, WAV_SAMPLE_RATE)
                    save_wav(record["answer_audio_path"], a_wav_ms, WAV_SAMPLE_RATE)
                    write_jsonl_record(jsonl_f, record)

                for writer in (
                    query_token_writer,
                    query_audio_writer,
                    answer_token_writer,
                    answer_audio_writer,
                ):
                    writer.flush()
                jsonl_f.flush()

        print("All samples processed.")
