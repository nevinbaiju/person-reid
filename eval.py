import torch
import torch.nn as nn
from ignite.engine import Engine

from reid_metric import R1_mAP

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    #     model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def inference(
        model,
        val_loader,
        num_query,
        early_stop=False
):
    device = "cuda"

    print("Enter inferencing")
    
    print("Create evaluator")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm='yes')}, 
                                            device=device)

    evaluator.run(val_loader)
    if early_stop:
        return 0,0
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    print('Validation Results')
    print("mAP: {:.1%}".format(mAP))
    cmc_result = ""
    for r in [1] + [x for x in range(2, 33, 2)]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc, mAP 