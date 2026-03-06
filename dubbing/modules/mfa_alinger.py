# Command line functions for aligning single files

from __future__ import annotations

from pathlib import Path

import pywrapfst
from kalpy.aligner import KalpyAligner
import torch
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler
from kalpy.utterance import Segment
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner import config
from montreal_forced_aligner.alignment import PretrainedAligner

from montreal_forced_aligner.corpus.classes import FileData, UtteranceData
from montreal_forced_aligner.data import (
    BRACKETED_WORD,
    CUTOFF_WORD,
    LAUGHTER_WORD,
    OOV_WORD,
    Language,
    TextFileType,
)
from montreal_forced_aligner.dictionary.mixins import (
    DEFAULT_BRACKETS,
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    DEFAULT_WORD_BREAK_MARKERS,
)
from montreal_forced_aligner.models import AcousticModel, G2PModel, DictionaryModel
from montreal_forced_aligner.online.alignment import tokenize_utterance_text
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer

import librosa
import tgt

class MFAAligner:
    def __init__(
        self,
        acoustic_model: AcousticModel | str = "english_us_arpa",
        dictionary_model: DictionaryModel | str = "english_us_arpa",
        config_path=None,
        g2p_model_path=None,
        no_tokenization=False,
    ):
        if isinstance(acoustic_model, str):
            self.acoustic_model_path = AcousticModel.get_pretrained_path(acoustic_model)
        self.acoustic_model = AcousticModel(self.acoustic_model_path)

        if isinstance(dictionary_model, str):
            self.dictionary_model_path = DictionaryModel.get_pretrained_path(dictionary_model)
        else:
            self.dictionary_model_path = None

        # ---- 预初始化与输入无关的静态组件，避免每次 align 重复加载 ----
        self._g2p_model = G2PModel(g2p_model_path) if g2p_model_path else None
        self._c = PretrainedAligner.parse_parameters(config_path)

        extracted_models_dir = config.TEMPORARY_DIRECTORY.joinpath("extracted_models", "dictionary")
        dictionary_directory = extracted_models_dir.joinpath(self.dictionary_model_path.stem)
        dictionary_directory.mkdir(parents=True, exist_ok=True)
        l_fst_path       = dictionary_directory.joinpath("L.fst")
        l_align_fst_path = dictionary_directory.joinpath("L_align.fst")
        words_path       = dictionary_directory.joinpath("words.txt")
        phones_path      = dictionary_directory.joinpath("phones.txt")

        lexicon_compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=self.acoustic_model.parameters["silence_probability"],
            initial_silence_probability=self.acoustic_model.parameters["initial_silence_probability"],
            final_silence_correction=self.acoustic_model.parameters["final_silence_correction"],
            final_non_silence_correction=self.acoustic_model.parameters["final_non_silence_correction"],
            silence_phone=self.acoustic_model.parameters["optional_silence_phone"],
            oov_phone=self.acoustic_model.parameters["oov_phone"],
            position_dependent_phones=self.acoustic_model.parameters["position_dependent_phones"],
            phones=self.acoustic_model.parameters["non_silence_phones"],
            ignore_case=self._c.get("ignore_case", True),
        )
        if l_fst_path.exists() and not config.CLEAN:
            lexicon_compiler.load_l_from_file(l_fst_path)
            lexicon_compiler.load_l_align_from_file(l_align_fst_path)
            lexicon_compiler.word_table = pywrapfst.SymbolTable.read_text(words_path)
            lexicon_compiler.phone_table = pywrapfst.SymbolTable.read_text(phones_path)
        else:
            lexicon_compiler.load_pronunciations(self.dictionary_model_path)
            lexicon_compiler.create_fsts()
            lexicon_compiler.clear()
            # 持久化编译结果，下次直接加载
            lexicon_compiler.fst.write(str(l_fst_path))
            lexicon_compiler.align_fst.write(str(l_align_fst_path))
            lexicon_compiler.word_table.write_text(words_path)
            lexicon_compiler.phone_table.write_text(phones_path)
        self._lexicon_compiler = lexicon_compiler

        if no_tokenization or self.acoustic_model.language is Language.unknown:
            self._tokenizer = SimpleTokenizer(
                word_table=lexicon_compiler.word_table,
                word_break_markers=self._c.get("word_break_markers", DEFAULT_WORD_BREAK_MARKERS),
                punctuation=self._c.get("punctuation", DEFAULT_PUNCTUATION),
                clitic_markers=self._c.get("clitic_markers", DEFAULT_CLITIC_MARKERS),
                compound_markers=self._c.get("compound_markers", DEFAULT_COMPOUND_MARKERS),
                brackets=self._c.get("brackets", DEFAULT_BRACKETS),
                laughter_word=self._c.get("laughter_word", LAUGHTER_WORD),
                oov_word=self._c.get("oov_word", OOV_WORD),
                bracketed_word=self._c.get("bracketed_word", BRACKETED_WORD),
                cutoff_word=self._c.get("cutoff_word", CUTOFF_WORD),
                ignore_case=self._c.get("ignore_case", True),
            )
        else:
            self._tokenizer = generate_language_tokenizer(self.acoustic_model.language)

        align_options = {
            k: v for k, v in self._c.items()
            if k in ["beam", "retry_beam", "acoustic_scale",
                     "transition_scale", "self_loop_scale", "boost_silence"]
        }
        self._kalpy_aligner = KalpyAligner(self.acoustic_model, lexicon_compiler, **align_options)
            
    def _build_utteranceData(self, 
                            text: str,
                            speaker_name: str = None,
                            file_name: str = None,
                            begin: float = 0.0,
                            end: float = None,
                             ):
        """
        Generate UtteranceData for alignment from text and other metadata.

        Args:
            text (str): _description_
            speaker_name (str, optional): _description_. Defaults to None.
            file_name (str, optional): _description_. Defaults to None.
            begin (float, optional): _description_. Defaults to 0.0.
            end (float, optional): _description_. Defaults to None.
        """
        uttdata = []
        utt = UtteranceData(
            text=text,
            speaker_name=speaker_name,
            file_name=file_name,
            begin=begin,
            end=end,
            channel=0,
        )
        
        uttdata.append(utt)
        
        return uttdata
    
    def _build_Segment(self,
                       wavs: torch.Tensor,
                       sampling_rate: int = 22050,
                       utterance: UtteranceData = None,
                       ):
        
        """Build a Segment object for alignment from wavs and utterance metadata.
        
        Args:
            wavs (torch.Tensor): The input wavs to be aligned.
            sampling_rate (int, optional): The sampling rate of the wavs. Defaults to
                22050.
            utterance (UtteranceData, optional): The metadata of the utterance. Defaults
                to None.
        """
        
        # resample wavs to 16kHz if needed
        if sampling_rate != 16000:
            wavs = wavs.cpu().numpy()
            wavs = librosa.resample(wavs, orig_sr=sampling_rate, target_sr=16000)
            
        if wavs.ndim > 1:
            wavs = wavs.mean(axis=0)
            
        seg = Segment(None, utterance.begin, utterance.end, utterance.channel)
        seg._wave = wavs
            
        return seg
        
    def align_one_file(
        self,
        sound_file_path: Path,
        text_file_path: Path,
        output_path: Path,
        **kwargs,
    ):
        """
        Align a single file with a pronunciation dictionary and a pretrained acoustic model.
        
        https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/command_line/align_one.py
        """
        config_path = kwargs.get("config_path", None)
        sound_file_path: Path = sound_file_path or kwargs["sound_file_path"]
        text_file_path: Path = text_file_path or kwargs["text_file_path"]
        dictionary_path: Path = self.dictionary_model_path or kwargs["dictionary_path"]
        acoustic_model_path = self.acoustic_model_path or kwargs["acoustic_model_path"]
        output_path: Path = output_path or kwargs["output_path"]
        if output_path.is_dir():
            output_path = output_path.joinpath(sound_file_path.stem + ".TextGrid")
        output_format = kwargs.get("output_format", "long_textgrid")
        no_tokenization = kwargs.get("no_tokenization", False)
        g2p_model_path = kwargs.get("g2p_model_path", None)

        acoustic_model = AcousticModel(acoustic_model_path)
        g2p_model = None
        if g2p_model_path:
            g2p_model = G2PModel(g2p_model_path)
        c = PretrainedAligner.parse_parameters(config_path)
        extracted_models_dir = config.TEMPORARY_DIRECTORY.joinpath("extracted_models", "dictionary")
        dictionary_directory = extracted_models_dir.joinpath(dictionary_path.stem)
        dictionary_directory.mkdir(parents=True, exist_ok=True)
        lexicon_compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=acoustic_model.parameters["silence_probability"],
            initial_silence_probability=acoustic_model.parameters["initial_silence_probability"],
            final_silence_correction=acoustic_model.parameters["final_silence_correction"],
            final_non_silence_correction=acoustic_model.parameters["final_non_silence_correction"],
            silence_phone=acoustic_model.parameters["optional_silence_phone"],
            oov_phone=acoustic_model.parameters["oov_phone"],
            position_dependent_phones=acoustic_model.parameters["position_dependent_phones"],
            phones=acoustic_model.parameters["non_silence_phones"],
            ignore_case=c.get("ignore_case", True),
        )
        l_fst_path = dictionary_directory.joinpath("L.fst")
        l_align_fst_path = dictionary_directory.joinpath("L_align.fst")
        words_path = dictionary_directory.joinpath("words.txt")
        phones_path = dictionary_directory.joinpath("phones.txt")
        if l_fst_path.exists() and not config.CLEAN:
            lexicon_compiler.load_l_from_file(l_fst_path)
            lexicon_compiler.load_l_align_from_file(l_align_fst_path)
            lexicon_compiler.word_table = pywrapfst.SymbolTable.read_text(words_path)
            lexicon_compiler.phone_table = pywrapfst.SymbolTable.read_text(phones_path)
        else:
            lexicon_compiler.load_pronunciations(dictionary_path)
            lexicon_compiler.create_fsts()
            lexicon_compiler.clear()

        if no_tokenization or acoustic_model.language is Language.unknown:
            tokenizer = SimpleTokenizer(
                word_table=lexicon_compiler.word_table,
                word_break_markers=c.get("word_break_markers", DEFAULT_WORD_BREAK_MARKERS),
                punctuation=c.get("punctuation", DEFAULT_PUNCTUATION),
                clitic_markers=c.get("clitic_markers", DEFAULT_CLITIC_MARKERS),
                compound_markers=c.get("compound_markers", DEFAULT_COMPOUND_MARKERS),
                brackets=c.get("brackets", DEFAULT_BRACKETS),
                laughter_word=c.get("laughter_word", LAUGHTER_WORD),
                oov_word=c.get("oov_word", OOV_WORD),
                bracketed_word=c.get("bracketed_word", BRACKETED_WORD),
                cutoff_word=c.get("cutoff_word", CUTOFF_WORD),
                ignore_case=c.get("ignore_case", True),
            )
        else:
            tokenizer = generate_language_tokenizer(acoustic_model.language)
        file_name = sound_file_path.stem
        file = FileData.parse_file(file_name, sound_file_path, text_file_path, "", 0)
        file_ctm = HierarchicalCtm([])
        utterances = []
        cmvn_computer = CmvnComputer()
        for utterance in file.utterances:
            seg = Segment(None, utterance.begin, utterance.end, utterance.channel)
            print(f"{utterance}")
            audio = librosa.load(sound_file_path, sr=16000,)[0]
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            seg._wave = audio
            normalized_text = tokenize_utterance_text(
                utterance.text,
                lexicon_compiler,
                tokenizer,
                g2p_model,
                language=acoustic_model.language,
            )
            utt = KalpyUtterance(seg, normalized_text)
            utt.generate_mfccs(acoustic_model.mfcc_computer)
            utterances.append(utt)

        cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs for utt in utterances])
        align_options = {
            k: v
            for k, v in c.items()
            if k
            in [
                "beam",
                "retry_beam",
                "acoustic_scale",
                "transition_scale",
                "self_loop_scale",
                "boost_silence",
            ]
        }
        if g2p_model is not None or not (l_fst_path.exists() and not config.CLEAN):
            lexicon_compiler.fst.write(str(l_fst_path))
            lexicon_compiler.align_fst.write(str(l_align_fst_path))
            lexicon_compiler.word_table.write_text(words_path)
            lexicon_compiler.phone_table.write_text(phones_path)
        kalpy_aligner = KalpyAligner(acoustic_model, lexicon_compiler, **align_options)
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm = kalpy_aligner.align_utterance(utt)
            file_ctm.word_intervals.extend(ctm.word_intervals)
        if str(output_path) != "-":
            output_path.parent.mkdir(parents=True, exist_ok=True)
        file_ctm.export_textgrid(
            output_path, file_duration=file.wav_info.duration, output_format=output_format
        )
        
    def align_one_wav(
        self,
        wavs: torch.Tensor = None,
        sampling_rate: int = 22050,
        text: str = None,
        return_textgrid: bool = True,
        **kwargs,
    ) -> HierarchicalCtm | tuple[tgt.TextGrid, list[list[tgt.Interval]]]:
        """
        Align a single wavs with a pronunciation dictionary and a pretrained acoustic model.
        所有与输入无关的组件（acoustic model, lexicon compiler, tokenizer, aligner）
        均在 __init__ 中预加载，此处只做逐帧特征提取与对齐。
        """
        wav_length_seconds = wavs.shape[-1] / sampling_rate

        raw_utterances_data = self._build_utteranceData(
            text=text,
            speaker_name=kwargs.get("speaker_name", None),
            file_name=kwargs.get("file_name", None),
            begin=0.0,
            end=wav_length_seconds,
        )

        file_ctm = HierarchicalCtm([])
        utterances = []
        cmvn_computer = CmvnComputer()
        for utterance in raw_utterances_data:
            seg = self._build_Segment(
                wavs=wavs,
                sampling_rate=sampling_rate,
                utterance=utterance,
            )
            normalized_text = tokenize_utterance_text(
                utterance.text,
                self._lexicon_compiler,
                self._tokenizer,
                self._g2p_model,
                language=self.acoustic_model.language,
            )
            utt = KalpyUtterance(seg, normalized_text)
            utt.generate_mfccs(self.acoustic_model.mfcc_computer)
            utterances.append(utt)

        cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs for utt in utterances])
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm = self._kalpy_aligner.align_utterance(utt)
            file_ctm.word_intervals.extend(ctm.word_intervals)

        if return_textgrid:
            return self.ctm_to_textgrid_fast(file_ctm)
        return file_ctm
        
    @staticmethod
    def ctm_to_textgrid_fast(
        h_ctm: HierarchicalCtm,
    ) -> tuple[tgt.TextGrid, list[list[tgt.Interval]]]:
        # 1. 一行代码搞定 words 层
        word_tier = tgt.IntervalTier(name='words', objects=[
            tgt.Interval(w.begin, w.end, w.label) for w in h_ctm.word_intervals
        ])

        # 2. 按 word 构建 phone groups (保留层级结构)
        phone_groups: list[list[tgt.Interval]] = [
            [tgt.Interval(p.begin, p.end, p.label) for p in w.phones]
            for w in h_ctm.word_intervals
        ]

        # 3. 展平 phone groups 得到 phones tier
        phone_tier = tgt.IntervalTier(name='phones', objects=[
            p for group in phone_groups for p in group
        ])

        # 4. 组装并导出
        tg = tgt.TextGrid()
        tg.add_tier(word_tier)
        tg.add_tier(phone_tier)

        return tg, phone_groups
            