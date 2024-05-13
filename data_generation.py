#%%
import json
import os
import numpy as np
import librosa
from pydub import AudioSegment
import random
from transformers import BertModel, BertTokenizer
import torch
#%%
# 正则化
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# hop_size 为窗口移动距离
# 窗口每秒移动30次，采样率为16000，经计算hop_size = 533

def convert_spectrogram(audio, frame_size=2048, hop_size=533):
    # extracting with Short-Time Fourier Transform
    S_scale = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
    spectrogram = np.abs(S_scale) ** 2
    # convert amplitude to DBs
    log_spectrogram = librosa.power_to_db(spectrogram)
    return log_spectrogram  # in dB
    # log_spectrogram.shape = [1025, audio/hop_size]


def convert_mel_spectrogram(
    audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80
):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=audio_sr,
        n_fft=frame_size,
        hop_length=hop_size,
        n_mels=num_mel_bands,
    )
    # mel-spectrogram.shape=[80, audio/hop_size]
    # convert amplitude to DBs
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # log-mel.shape=[80, audio/hop_size]
    return log_mel_spectrogram  # in dB


def get_num_frame(data, frame_size, hop_size):
    T = data.shape[-1]
    if T <= frame_size:
        return 1
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame


def get_text_hop_size(text, frame_size, num_frame):
    if num_frame == 1:
        return 0
    T = text.shape[0]
    return (T - frame_size) // (num_frame - 1)


def audio_padding(data, pad_size):
    if data.shape[1] != pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, : data.shape[1]] = data
    else:
        padded_data = data

    return padded_data


def text_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[: data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def sliding_window(
    spectro,
    mel_spectro,
    text_feature,
    sr,
    window_size,
    overlap_size,
    output_root,
    user_dir
):
    """
    sr是frame rate
    windows-size = 60 seconds
    hop-size
    因为overlap所以有了3条数据
    文字跟音频是要按照速率来映射的

    """
    frame_size = window_size * sr  # 60 * 30
    hop_size = (window_size - overlap_size) * sr  # (60 - 10) * 30
    num_frame = get_num_frame(spectro, frame_size, hop_size)

    text_frame_size = 175  # 500 samples: 76489 / 26169 * 60 = 175
    text_hop_size = max(0, get_text_hop_size(text_feature, text_frame_size, num_frame))

    segments = {
        "num_frame": num_frame,
        "spectrogram": [],
        "mel-spectrogram": [],
        "sentence_embeddings": [],
    }

    os.makedirs(os.path.join(output_root, user_dir, "spectrogram"))
    os.makedirs(os.path.join(output_root, user_dir, "mel-spectrogram"))
    os.makedirs(os.path.join(output_root, user_dir, "sentence_embeddings"))
    # start sliding through and generating data
    for i in range(num_frame):
        frame_sample_spec = audio_padding(
            spectro[:, i * hop_size : i * hop_size + frame_size], frame_size
        )
        frame_sample_mspec = audio_padding(
            mel_spectro[:, i * hop_size : i * hop_size + frame_size], frame_size
        )
        frame_sample_text = text_padding(
            text_feature[i * text_hop_size : i * text_hop_size + text_frame_size],
            text_frame_size,
        )

        # start storing
        np.save(
            os.path.join(output_root, user_dir,"spectrogram", f"{i}.npy"),
            frame_sample_spec,
        )
        np.save(
            os.path.join(output_root, user_dir,"mel-spectrogram", f"{i}.npy"),
            frame_sample_mspec,
        )
        np.save(
            os.path.join(output_root, user_dir,"sentence_embeddings", f"{i}.npy"),
            frame_sample_text,
        )
        segments["spectrogram"].append(os.path.join(user_dir, "spectrogram", f"{i}.npy"))
        segments["mel-spectrogram"].append(
            os.path.join(user_dir, "mel-spectrogram", f"{i}.npy")
        )
        segments["sentence_embeddings"].append(
            os.path.join(user_dir, "sentence_embeddings", f"{i}.npy")
        )

    return segments


def data2vec(data):
    audio_dir = r"./audio_output"
    output_dir = r"./data/np_data"
    model_path = r"./model/chinese-bert-wwm-ext"

    # model = SentenceTransformer(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    np_list = []
    for user in data:
        wav_dir = sorted(
            user.get("audios").items(),
            key= lambda item: int(item[0].split("_")[-1].split(".")[0])
        )
        user_id = user.get("user_id")
        audios_info = [
            x for x in wav_dir
            if os.path.exists(os.path.join(audio_dir, user_id, x[0]))
        ]

        # text feature
        # text_feature = model.encode([x[1] for x in audios_info])
        text = "".join([x[1] for x in audios_info])
        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        text_feature = model_output[0][0].numpy()

        # audio feature
        audio = None
        for item in audios_info:
            if not audio:
                audio = AudioSegment.from_file(
                    os.path.join(audio_dir, user_id, item[0]), format="wav"
                )
            else:
                audio += AudioSegment.from_file(
                    os.path.join(audio_dir, user_id, item[0]), format="wav"
                )
        if audio.frame_rate != 16000 or audio.channels != 1:
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)

        wav_array = np.asarray(audio.get_array_of_samples(), dtype=float)
        wav_sr = audio.frame_rate
        spectro = normalize(
            convert_spectrogram(wav_array, frame_size=2048, hop_size=533)
        )
        mel_spectro = normalize(
            convert_mel_spectrogram(
                wav_array,
                wav_sr,
                frame_size=2048,
                hop_size=533,
                num_mel_bands=80,
            ))

        # creating data
        sr = 30  # 30Hz
        window_size = 60  # 60s
        overlap_size = 10  # 10s
        segments = sliding_window(
            spectro,
            mel_spectro,
            text_feature,
            sr,
            window_size,
            overlap_size,
            output_dir,
            user_id
        )
        np_list.append({user_id: segments})

    json.dump(
        np_list,
        open(os.path.join(output_dir, "np_data.json"), "w", encoding="utf8"),
        ensure_ascii=False,
        indent=4,
    )
    return np_list


if __name__ == "__main__":
    output_dir = "./data/np_data"

    data = json.load(open("./data/wav_select.json", encoding="utf8"))
    np_list = data2vec(data)

# %%
