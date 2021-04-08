import os
import librosa
import math
import json

DATASET_PATH = "../gtzan_dataset/genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050  # Hz
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    '''
    Dictionary
    "mapping": ['classical', 'blues'],
    "mfcc": [[...], [...], [...]],
    "labels": [0,  0, 1]
    '''
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length)  # 1.2 -> 2

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we're not at the root level
        if dirpath is not dataset_path:
            # save semantic label
            dirpath_components = os.path.split(dirpath)
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load audio
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments for mfcc and storing data
                for s in range(num_segments):
                    start = num_samples_per_segment * s  # s=0 -> 0
                    # s=0 -> num_samples_per_segment
                    finish = start + num_samples_per_segment

                    # store mfcc for segment if it has expected length
                    mfcc = librosa.feature.mfcc(signal[start:finish],
                                                sr,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                n_mfcc=n_mfcc)


                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
