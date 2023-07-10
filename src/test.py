import cv2
import torch
import hydra
import numpy as np
import pyrootutils

from tqdm import tqdm
from dvclive import Live
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import F1Score

pyrootutils.setup_root(__file__, indicator='.project', pythonpath=True)

from src.utils import make_deterministic, get_device, make_dir
from src.model import make_anchors, ModelYOLO, DFL, dist2bbox
from src.dataset import get_datasets
from src.metrics import non_max_suppression


def load_model(model, checkpoint):
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.eval()
    return model


@hydra.main(version_base='1.3', config_path='../conf', config_name='config.yaml')
def main(cfg):
    params = cfg.test

    reg_max = 16
    number_of_classes = len(params.labels)
    no = number_of_classes + reg_max * 4

    make_deterministic(params.seed)
    device = get_device(params.device)

    model = ModelYOLO(nc=number_of_classes)
    model = load_model(model, params.model).to(device)

    _, test_dataset = get_datasets(params)

    loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False
    )

    anchors, strides = make_anchors(device)
    dfl = DFL(reg_max).to(device)

    with Live(dir='dvclive/test', save_dvc_exp=False) as live:
        #
        predictions = []
        targets = []

        target_labels = []
        prediction_labels = []

        image_counter = 0
        output_dir = make_dir(params.output_dir)

        for batch, targs in tqdm(loader, leave=False):
            batch = batch.float().unsqueeze(1).to(device) / 255
            batch_size = batch.shape[0]

            with torch.no_grad():
                preds = model(batch)

            preds = [p.view(batch_size, no, -1) for p in preds]
            box, cls = torch.cat(preds, 2).split((reg_max * 4, number_of_classes), 1)
            dbox = dist2bbox(dfl(box), anchors, xywh=False, dim=1) * strides

            y = torch.cat((dbox, cls.sigmoid()), 1)
            nms_y = non_max_suppression(y, nc=number_of_classes)

            for t, p in zip(targs, nms_y):
                if len(p) == 0:
                    continue

                p = p[0].cpu()

                bbox = torch.tensor([[t[1], t[2], t[3], t[4]]])
                label = torch.tensor([t[0]]).long()

                targets.append(dict(boxes=bbox, labels=label))

                predictions.append(
                    dict(boxes=torch.tensor([[p[0], p[1], p[2], p[3]]]),
                         scores=torch.tensor([p[4]]),
                         labels=torch.tensor([p[5]]).long())
                )

                target_labels.append(t[0])
                prediction_labels.append(p[5])

                # save inference
                for b, t, p in zip(batch, targs, nms_y):
                    label, x1, y1, x2, y2 = t.numpy().astype(int)

                    image = (255 * b[0].cpu().numpy()).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if len(p) > 0:
                        x1, y1, x2, y2, conf, cls = p[0].cpu().numpy().astype(int)
                        h, w = image.shape[:2]

                        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # B G R

                    file = output_dir / (str(image_counter).zfill(6) + '.jpg')
                    cv2.imwrite(str(file), image)
                    image_counter += 1

        mean_ap = MeanAveragePrecision()
        mean_ap.update(predictions, targets)

        f1_score = F1Score(task='multiclass', num_classes=number_of_classes)
        f1_score_value = f1_score(torch.asarray(prediction_labels), torch.asarray(target_labels))

        live.log_metric('mAP@50', mean_ap.compute()['map_50'].item())
        live.log_metric('F1score', f1_score_value.item())


if __name__ == '__main__':
    main()
