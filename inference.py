import pickle
import torch
import numpy as np

from model.ctrgcn import Model


class CTRGCNPredictor:

    def __init__(self):

        self.model = Model(
            num_class=1000,
            num_point=75,
            num_person=1,
            graph="graph.mediapipe75.Graph",
            graph_args={
                "labeling_mode": "spatial"
            }
        )

        print("Loading CTR-GCN...")

        weights = torch.load(
            "work_dir/wlasl1000/ctrgcn/runs-30-11220.pt",
            map_location="cpu"
        )

        self.model.load_state_dict(
            weights
        )

        self.model.eval()

        with open(
            "class_to_word.pkl",
            "rb"
        ) as f:

            self.class_map = pickle.load(f)

        print(
            f"Loaded {len(self.class_map)} labels"
        )

    # -----------------------------------
    # NEW FUNCTION
    # -----------------------------------

    def predict_word(
        self,
        sequence
    ):

        results = self.predict(
            sequence
        )

        if len(results) == 0:
            return "Unknown"

        return results[0]["word"]
    
    def predict_word_with_confidence(self,sequence):
        results = self.predict(sequence)
        if len(results) == 0:
            return ("Unknown",0 )
        return (
        results[0]["word"],
        results[0]["confidence"]
    )

    # -----------------------------------
    # EXISTING FUNCTION
    # -----------------------------------

    def predict(self, sequence):

        if len(sequence) == 0:
            return []

        data = np.array(
            sequence,
            dtype=np.float32
        )

        if data.shape[0] != 32:
            return []

        data = np.transpose(
            data,
            (2, 0, 1)
        )

        data = data.reshape(
            1,
            3,
            32,
            75,
            1
        )

        x = torch.tensor(
            data,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = self.model(x)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            top5_prob, top5_idx = torch.topk(
                probabilities,
                5
            )

        results = []

        for i in range(5):

            class_id = (
                top5_idx[0][i]
                .item()
            )

            confidence = (
                top5_prob[0][i]
                .item() * 100
            )

            word = self.class_map.get(
                class_id,
                f"class_{class_id}"
            )

            results.append(
                {
                    "class_id": class_id,
                    "word": word,
                    "confidence": confidence
                }
            )

        return results