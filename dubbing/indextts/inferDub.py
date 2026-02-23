from indextts.infer_v2 import IndexTTS2
from typing import Any, List
from dataclasses import dataclass

import torch
import torchaudio
import time
import os
import warnings
import random


@dataclass
class IndexTTS2Outputs:
    wavs: torch.Tensor = None
    vc_targets: torch.Tensor = None
    seg_lens: List[int] = None
    wav_length: float = None
    inference_stats: dict | None = None
    
    sampling_rate: int = 22050


class IndexTTS2ForDub(IndexTTS2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Initializing IndexTTS2ForDub...")
        self.logger.info(f"IndexTTS2ForDub initialized with args: {args}, kwargs: {kwargs}")
        
    def infer_dub(
        self, 
        spk_audio_prompt: str = None,
        text: List[str] | str = None, 
        output_path: str = None,
        emo_audio_prompt: list[str] | str = None,
        emo_alpha: float = 1.0,
        emo_vector: list = None,
        style_prompt: Any = None,
        use_emo_text: bool = False,
        emo_text: List[str] | str = None, 
        use_random: bool = False,
        interval_silence: float = 200,
        verbose: bool = False,
        max_text_tokens_per_sentence: int = 120,
        return_stats: bool = False,
        save_attention_maps: bool = False,
        **generation_kwargs):
        """
        Run dubbing-oriented speech synthesis using speaker/emotion prompts.

        Args:
            spk_audio_prompt (str | list[str], optional): Speaker reference audio path(s).
                A single path conditions one voice; multiple paths are fused for mixed reference.
            text (str | list[str], optional): Text to synthesize. Current implementation expects
                a string and uses "|" as segment separators for multi-part dubbing.
            output_path (str, optional): Target path to save the generated waveform. If None,
                returns audio data in Gradio-compatible tuple format.
            emo_audio_prompt (str | list[str], optional): Emotion reference audio path(s). If None,
                speaker reference audio is reused as emotion reference.
            emo_alpha (float, optional): Blend factor used when merging speaker and emotion
                conditioning embeddings.
            emo_vector (list, optional): Explicit emotion vector(s). When provided, this overrides
                `emo_audio_prompt` and forces emotion conditioning from vectors.
            style_prompt (Any, optional): Optional style reference used when multiple speaker prompts
                are provided; otherwise averaged style is used.
            use_emo_text (bool, optional): Whether to infer emotion vectors from text via
                `QwenEmotion`.
            emo_text (str | list[str], optional): Text used for emotion analysis when
                `use_emo_text=True`. If None, synthesis `text` is used.
            use_random (bool, optional): Whether to sample random emotion prototypes from the
                emotion matrix instead of nearest-speaker matching.
            interval_silence (float, optional): Silence duration (milliseconds) inserted between
                generated segments.
            verbose (bool, optional): Whether to print detailed runtime/debug logs.
            max_text_tokens_per_sentence (int, optional): Soft limit used in warnings and sentence
                control for long text generation.
            return_stats (bool, optional): Whether to return detailed inference statistics in
                addition to audio output.
            save_attention_maps (bool, optional): Whether to save GPT attention maps during
                generation.
            **generation_kwargs: Extra generation arguments forwarded to GPT decoding.

                Returns:
                        IndexTTS2Outputs: Wrapped synthesis outputs.
                                - `wavs`: generated waveform tensor.
                                - `vc_targets`: generated mel target tensor.
                                - `seg_lens`: segment lengths inferred by GPT alignment.
                                - `wav_length`: waveform duration in seconds.
                                - `inference_stats`: detailed stats when `return_stats=True`, otherwise `None`.
        """
        
        self.logger.stage("Starting Inference")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            self.logger.print_dict("Inference Parameters", {
                "text": text,
                "spk_audio_prompt": spk_audio_prompt,
                "emo_audio_prompt": emo_audio_prompt,
                "emo_alpha": emo_alpha,
                "emo_vector": emo_vector,
                "use_emo_text": use_emo_text,
                "emo_text": emo_text
            })
        start_time = time.perf_counter()
        emo_vectors = emo_vector
        emo_alpha = max(0.0, min(emo_alpha, 1.4))  # ensure emo_alpha is in [0, 1]
        if emo_alpha > 1.0:
            self.logger.warning(f"Warning: emo_alpha {emo_alpha} is greater than 1.0, which may lead to exaggerated emotion effects. ")
        
        if use_emo_text:
            emo_vectors = []
            emo_audio_prompt = None
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0
            
            # if emo_text is None, use synthesis text as emotion analysis text by default
            if emo_text is None:
                emo_text = text
                
            if isinstance(emo_text, str):
                emo_texts = emo_text.split("|")
            else:
                emo_texts = emo_text
                
            for emo_text in emo_texts:
                emo_dict = self.qwen_emo.inference(emo_text)
                # convert ordered dict to list of vectors; the order is VERY important!
                emo_vector = list(emo_dict.values())
                
                # scale emo_vector by emo_alpha
                emo_vector = [val * emo_alpha for val in emo_vector]
                emo_vectors.append(emo_vector)
        # emo_vectors = [[1., 0, 0, 0, 0, 0, 0, 0], emo_vector, [1., 0, 0, 0, 0, 0, 0, 0]]
        # emo_vectors = [emo_vector]
        if emo_vectors is not None:
            emo_audio_prompt = None
            EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
            # Collect all emotion vectors for display
            emo_vec_maps = []
            for emo in emo_vectors:
                emo_vec_map = {dim: val for dim, val in zip(EMOTION_DIMENSIONS, emo)}
                emo_vec_maps.append(emo_vec_map)
            # Display all emotion vectors side by side
            self.logger.print_emotion_vector(emo_vec_maps)
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            # assert emo_alpha == 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            spk_cond_emb_list = []
            self._set_gr_progress(0.05, "processing reference audio...")
            self.logger.info("Processing reference audio...")
            if isinstance(spk_audio_prompt, str):
                audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
                audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
                audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

                inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
                input_features = inputs["input_features"]
                attention_mask = inputs["attention_mask"]
                input_features = input_features.to(self.device)
                attention_mask = attention_mask.to(self.device)
                spk_cond_emb = self.get_emb(input_features, attention_mask)

                _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
                ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
                ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
                feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                        num_mel_bins=80,
                                                        dither=0,
                                                        sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
                style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

                prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                        ylens=ref_target_lengths,
                                                                        n_quantizers=3,
                                                                        f0=None)[0]
            elif isinstance(spk_audio_prompt, list):
                ref_mel_list = []
                styles_list = []
                prompt_conditions_list = []
                for spk_path in spk_audio_prompt:
                    audio, sr = self._load_and_cut_audio(spk_path, 15, verbose)
                    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
                    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

                    inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs["input_features"]
                    attention_mask = inputs["attention_mask"]
                    input_features = input_features.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    spk_cond_emb = self.get_emb(input_features, attention_mask)
                    spk_cond_emb_list.append(spk_cond_emb)

                    _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
                    ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
                    ref_mel_list.append(ref_mel)
                    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
                    feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                            num_mel_bins=80,
                                                            dither=0,
                                                            sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
                    style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]
                    styles_list.append(style)

                    prompt_condition_tmp = self.s2mel.models['length_regulator'](S_ref,
                                                                                ylens=ref_target_lengths,
                                                                                n_quantizers=3,
                                                                                f0=None)[0]
                    prompt_conditions_list.append(prompt_condition_tmp)
                    
                spk_cond_emb = spk_cond_emb_list
                ref_mel = torch.cat(ref_mel_list, dim=2)
                prompt_condition = torch.cat(prompt_conditions_list, dim=1)
                
                # ref_mel = ref_mel_list[0]
                # prompt_condition = prompt_conditions_list[0]
                if style_prompt is not None:
                    style_audio, sr = self._load_and_cut_audio(style_prompt, 15, verbose)
                    style_audio_16k = torchaudio.transforms.Resample(sr, 16000)(style_audio)
                    
                    feat = torchaudio.compliance.kaldi.fbank(style_audio_16k.to(ref_mel.device),
                                                            num_mel_bins=80,
                                                            dither=0,
                                                            sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)
                    style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]
                else:
                    style = torch.stack(styles_list, dim=0).mean(dim=0)
                    # style = styles_list[0]
                
            else:
                raise ValueError("`spk_audio_prompt` must be str or list.")
                
            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel
        
        # 新增模块：情感矩阵
        emovec_mats = []
        emo_weight_sums = None
        if emo_vectors is not None and len(emo_vectors) > 0:
            emo_vectors_tensor = torch.as_tensor(emo_vectors, device=self.device, dtype=style.dtype)
            emo_weight_sums = emo_vectors_tensor.sum(dim=1, keepdim=True)

            if use_random:
                for emo_vector in emo_vectors_tensor:
                    random_index = [random.randint(0, x - 1) for x in self.emo_num]
                    emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                    emo_matrix = torch.cat(emo_matrix, 0)
                    emovec_mat = torch.sum(emo_vector.unsqueeze(1) * emo_matrix, 0, keepdim=True)
                    emovec_mats.append(emovec_mat)
            else:
                random_index = [self.find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
                emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                emo_matrix = torch.cat(emo_matrix, 0)
                emovec_mats_tensor = emo_vectors_tensor @ emo_matrix
                emovec_mats = list(emovec_mats_tensor.split(1, dim=0))

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            self._set_gr_progress(0.08, "processing emotion audio...")
            self.logger.info("Processing emotion reference...")
            if isinstance(emo_audio_prompt, str):
                emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
                emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                emo_input_features = emo_inputs["input_features"]
                emo_attention_mask = emo_inputs["attention_mask"]
                emo_input_features = emo_input_features.to(self.device)
                emo_attention_mask = emo_attention_mask.to(self.device)
                emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

                self.cache_emo_cond = emo_cond_emb
                self.cache_emo_audio_prompt = emo_audio_prompt
            elif isinstance(emo_audio_prompt, list):
                emo_cond_embs = []
                for emo_audio_path in emo_audio_prompt:
                    emo_audio, _ = self._load_and_cut_audio(emo_audio_path, 15, verbose, sr=16000)
                    emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                    emo_input_features = emo_inputs["input_features"]
                    emo_attention_mask = emo_inputs["attention_mask"]
                    emo_input_features = emo_input_features.to(self.device)
                    emo_attention_mask = emo_attention_mask.to(self.device)
                    emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
                    emo_cond_embs.append(emo_cond_emb)
                
                self.cache_emo_cond = emo_cond_embs
                self.cache_emo_audio_prompt = emo_audio_prompt
            else:
                raise ValueError("emo_audio_prompt must be str or list of str")
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")        
        self.logger.info("Processing text...")        # 以“|”作为句子分割符，不同部分对应不同的情绪
        
        if isinstance(text, str):
            text_list = text.split("|")
        else:
            text_list = text
        text_tokens_list = []
        for txt in text_list:
            text_tokens_list.append(self.tokenizer.tokenize(txt))
        # sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        target_duration_tokens = generation_kwargs.pop("target_duration_tokens", None)
        sampling_rate = 22050

        if output_path is not None and "output_path" not in generation_kwargs:
            generation_kwargs["output_path"] = output_path
        
        generation_kwargs["save_attention_maps"] = save_attention_maps

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        # for sent in sentences:


        text_tokens_list = [self.tokenizer.convert_tokens_to_ids(sent) for sent in text_tokens_list]
        text_tokens_list = [torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0) for text_tokens in text_tokens_list]
        
        self._set_gr_progress(0.15, "generating speech codes...")        
        self.logger.info("Generating speech codes with GPT...")

        m_start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=self.dtype is not None, dtype=self.dtype):
                emovecs = []
                spk_cond_lengths_list = []
                emo_cond_lengths_list = []

                if isinstance(spk_audio_prompt, list):
                    for spk_cond_emb_tmp, emo_cond_emb_tmp in zip(spk_cond_emb_list, emo_cond_embs):
                        spk_cond_lengths = torch.tensor([spk_cond_emb_tmp.shape[-1]], device=text_tokens_list[0].device)
                        emo_cond_lengths = torch.tensor([emo_cond_emb_tmp.shape[-1]], device=text_tokens_list[0].device)
                        emovec = self.gpt.merge_emovec(
                            spk_cond_emb_tmp,
                            emo_cond_emb_tmp,
                            spk_cond_lengths,
                            emo_cond_lengths,
                            alpha=emo_alpha
                        )
                        emovecs.append(emovec)
                        spk_cond_lengths_list.append(spk_cond_lengths)
                        emo_cond_lengths_list.append(emo_cond_lengths)
                else:
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens_list[0].device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens_list[0].device),
                        alpha=emo_alpha
                    )
                    
                if emo_vectors is not None and len(emo_vectors) > 0:
                    if len(emovec_mats) > 0:
                        base_emovec = emovecs[-1] if len(emovecs) > 0 else emovec
                        emovec_mats_tensor = torch.cat(emovec_mats, dim=0)
                        if emo_weight_sums is None:
                            emo_weight_sums = torch.ones(
                                (emovec_mats_tensor.size(0), 1),
                                device=emovec_mats_tensor.device,
                                dtype=emovec_mats_tensor.dtype,
                            )
                        mixed_emovecs = emovec_mats_tensor + (1 - emo_weight_sums.to(base_emovec.dtype)) * base_emovec.expand(emovec_mats_tensor.size(0), -1)
                        emovecs.extend(mixed_emovecs.split(1, dim=0))
                elif len(emovecs) == 0:
                    emovecs.append(emovec)
                
                if len(spk_cond_lengths_list) > 0 and len(emo_cond_lengths_list) > 0:
                    input_cond_lengths = spk_cond_lengths_list
                    input_emo_cond_lengths = emo_cond_lengths_list
                else:
                    input_cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens_list[0].device)
                    input_emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens_list[0].device)
                
                # print(f"Merge emovec time: {time.perf_counter() - p_start_time:.2f}s")

                gpt_output = self.gpt.inference_speech(
                    spk_cond_emb,
                    text_tokens_list,
                    emo_cond_emb,
                    cond_lengths=input_cond_lengths,
                    emo_cond_lengths=input_emo_cond_lengths,
                    emo_vecs=emovecs,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    target_duration_tokens=target_duration_tokens,
                    **generation_kwargs
                )
                
                codes, speech_conditioning_latent, \
                    attention_mask, seg_lens, token_generation_time, aligned_sequences = gpt_output

            gpt_gen_time += time.perf_counter() - m_start_time
            self._set_gr_progress(0.4, "speech codes generated")            
            self.logger.success(f"Speech codes generated in {gpt_gen_time:.2f}s")
            if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                warnings.warn(
                    f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                    f"Input text tokens: {text_tokens_list[0].shape[1]}. "
                    f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                    category=RuntimeWarning
                )
                has_warned = True

            code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
            #                 if verbose:
            #                     print(codes, type(codes))
            #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
            #                     print(f"code len: {code_lens}")

            code_lens = []
            for code in codes:
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                    code_len = len_ - 1
                code_lens.append(code_len)
            codes = codes[:, :code_len]
            code_lens = torch.LongTensor(code_lens)
            code_lens = code_lens.to(self.device)
            if verbose:
                self.logger.debug(f"Codes: {codes}, type: {type(codes)}")
                self.logger.debug(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                self.logger.debug(f"code len: {code_lens}")

            self._set_gr_progress(0.45, "computing speech latent...")
            self.logger.info("Computing speech latent with GPT forward...")
            m_start_time = time.perf_counter()
            
            if isinstance(spk_cond_emb, list):
                bs = spk_cond_emb[0]
            else:
                bs = spk_cond_emb
            use_speed = torch.zeros(bs.size(0)).to(bs.device).long()
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=self.dtype is not None, dtype=self.dtype):
                latent = self.gpt(
                    speech_conditioning_latent,
                    torch.cat(text_tokens_list, dim=1),
                    torch.tensor([text_tokens_list[0].shape[-1]], device=text_tokens_list[0].device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=text_tokens_list[0].device),
                    emo_cond_emb,
                    cond_mel_lengths=input_cond_lengths,
                    emo_cond_mel_lengths=input_emo_cond_lengths,
                    emo_vecs=emovecs,
                    use_speed=use_speed,
                    attention_mask=attention_mask,
                )
                gpt_forward_time += time.perf_counter() - m_start_time
                self._set_gr_progress(0.55, "generating mel-spectrogram...")
                self.logger.info("Generating mel-spectrogram with S2Mel...")

            dtype = None
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=dtype is not None, dtype=dtype):
                m_start_time = time.perf_counter()
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = self.s2mel.models['gpt_layer'](latent)
                S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                
                mel_ratio = 1 / 50 * 22050 / 256
                target_lengths = (code_lens * mel_ratio).long()
                
                # speed_list = [0.8, 1.6, 2.5]
                # speech_conditions_list = []
                # prev_seg = 0
                # for i, seg in enumerate(seg_lens):
                #     curr_seg = prev_seg + seg
                #     part_S_infer = S_infer[:, prev_seg:curr_seg, :]
                #     part_target_length = part_S_infer.size(1) * 1.72 * speed_list[i % len(speed_list)]
                #     part_target_length = torch.tensor([part_target_length], device=part_S_infer.device).long()
                #     speech_conds = self.s2mel.models['length_regulator'](part_S_infer,
                #                                                 ylens=part_target_length,
                #                                                 n_quantizers=3,
                #                                                 f0=None)[0]
                #     speech_conditions_list.append(speech_conds)
                #     prev_seg = curr_seg
                # self.decode_text(
                #     text_tokens_list=text_tokens_list,
                #     aligned_sequences=aligned_sequences,
                #     output_wav_path=output_path,
                # )

                cond = self.s2mel.models['length_regulator'](S_infer,
                                                                ylens=target_lengths,
                                                                n_quantizers=3,
                                                                f0=None)[0]
                
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                print(f"cat_condition shape: {cat_condition.shape}")
                print(f"ref_mel shape: {ref_mel.shape}")
                vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                torch.LongTensor([cat_condition.size(1)]).to(
                                                                    cat_condition.device),
                                                                ref_mel, style, None, diffusion_steps,
                                                                inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_time += time.perf_counter() - m_start_time
                self._set_gr_progress(0.75, "synthesizing waveform...")
                self.logger.info("Synthesizing waveform with BigVGAN...")

                m_start_time = time.perf_counter()
                wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                if verbose:
                    self.logger.debug(f"BigVGAN output shape: {wav.shape}")
                bigvgan_time += time.perf_counter() - m_start_time
                wav = wav.squeeze(1)

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            if verbose:
                self.logger.debug(f"wav shape: {wav.shape}, min: {wav.min():.4f}, max: {wav.max():.4f}")
            # wavs.append(wav[:, :-512])
            wavs.append(wav.cpu())  # to cpu before saving

        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        total_inference_time = end_time - start_time
        rtf = total_inference_time / wav_length if wav_length > 0 else float("inf")
        
        # Print time statistics with beautiful table
        time_stats = {
            "GPT Generation": gpt_gen_time,
            "GPT Forward": gpt_forward_time,
            "S2Mel": s2mel_time,
            "BigVGAN": bigvgan_time,
        }
        
        inference_stats = {
            "total_inference_time": total_inference_time,
            "wav_length": wav_length,
            "rtf": rtf,
            "token_generation_time": token_generation_time,
            "gpt_gen_time": gpt_gen_time,
            "gpt_forward_time": gpt_forward_time,
            "s2mel_time": s2mel_time,
            "bigvgan_time": bigvgan_time,
            "S_infer": S_infer.cpu(),
        }
        self.logger.print_time_stats(time_stats, total_inference_time, wav_length)

        # save audio
        wav = wav.cpu()  # to cpu
        
        return IndexTTS2Outputs(
            wavs=wav,
            vc_targets=vc_target,
            seg_lens=seg_lens,
            wav_length=wav_length,
            inference_stats=inference_stats if return_stats else None,
            sampling_rate=sampling_rate,
        )
        

