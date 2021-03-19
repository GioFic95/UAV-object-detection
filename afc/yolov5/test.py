import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         log_imgs=0,  # number of logged images
         compute_loss=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    check_dataset(data)  # check
    nc1 = 1 if single_cls else int(data['nc1'])  # number of classes  # edit
    nc2 = 1 if single_cls else int(data['nc2'])  # number of classes  # edit
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True,
                                       prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]

    seen = 0
    confusion_matrix1 = ConfusionMatrix(nc=nc1)
    confusion_matrix2 = ConfusionMatrix(nc=nc2)  # edit
    names1 = {k: v for k, v in enumerate(model.names1 if hasattr(model, 'names1') else model.module.names1)}  # edit
    names2 = {k: v for k, v in enumerate(model.names2 if hasattr(model, 'names2') else model.module.names2)}  # edit
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    t0, t1 = 0., 0.
    p_1, r_1, f1_1, mp_1, mr_1, map50_1, map_1 = 0., 0., 0., 0., 0., 0., 0.
    p_2, r_2, f1_2, mp_2, mr_2, map50_2, map_2 = 0., 0., 0., 0., 0., 0., 0.  # edit
    loss = torch.zeros(4, device=device)  # edit
    jdict, stats1, stats2, ap_1, ap_2, ap50_1, ap50_2, ap_class_1, ap_class_2, wandb_images =\
        [], [], [], [], [], [], [], [], [], []  # edit

    # targets: img_id, cls1, cls2, xywh  # edit
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # print("targets 105:", targets.shape, targets)  # todo
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                # print("new_loss:", new_loss.shape)  # todo
                loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # box, obj, cls1, cls2  # edit

            # Run NMS
            targets[:, 3:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels  # edit
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, nc1=nc1, nc2=nc2)  # edit
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):  # pred: xyxy, conf1, cls1, conf2, cls2  # edit
            labels = targets[targets[:, 0] == si, 1:]  # labels: cls1, cls2, xywh  # edit
            # print("labels 130:", targets.shape, pred.shape, labels.shape, targets, labels)  # todo
            nl = len(labels)
            tcls1 = labels[:, 0].tolist() if nl else []  # target class 1
            tcls2 = labels[:, 1].tolist() if nl else []  # target class 2  # edit
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats1.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls1))
                    stats2.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls2))  # edit
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf1, cls1, conf2, cls2 in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls1, cls2, *xywh, conf1, conf2) if save_conf else (cls1, cls2, *xywh)  # label format  # edit
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging  # edit
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls1),
                             "box_caption": "%s (%s) %.3f" % (names1[cls1], names2[cls2], conf1),
                             "scores": {"class_score": conf1},
                             "domain": "pixel"} for *xyxy, conf1, cls1, conf2, cls2 in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names1}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls2),
                             "box_caption": "(%s) %s %.3f" % (names1[cls1], names2[cls2], conf2),
                             "scores": {"class_score": conf2},
                             "domain": "pixel"} for *xyxy, conf1, cls1, conf2, cls2 in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names2}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if save_json:  # todo ~
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct1 = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            correct2 = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)  # edit
            if nl:
                detected = []  # target indices
                tcls_tensor_1 = labels[:, 0]
                tcls_tensor_2 = labels[:, 1]  # edit

                # target boxes
                tbox = xywh2xyxy(labels[:, 2:6])  # edit
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix1.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))  # edit
                    confusion_matrix2.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))  # edit

                # Per target class
                unique_classes = torch.unique(torch.stack((tcls_tensor_1, tcls_tensor_2)), dim=1).T
                # print("unique_classes", unique_classes.shape, unique_classes)   # todo
                # print("tcls tensors", tcls_tensor_1, tcls_tensor_2)  # todo
                for cls1, cls2 in unique_classes:
                    ti1 = (cls1 == tcls_tensor_1).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi1 = (cls1 == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    ti2 = (cls2 == tcls_tensor_2).nonzero(as_tuple=False).view(-1)  # prediction indices  #edit
                    pi2 = (cls2 == pred[:, 7]).nonzero(as_tuple=False).view(-1)  # target indices  #edit
                    # print("ti/pi", ti1, pi1, ti2, pi2)  # todo

                    # Search for detections
                    if pi1.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi1, :4], tbox[ti1]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti1[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct1[pi1[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
                    if pi2.shape[0]:  # edit
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi2, :4], tbox[ti2]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti2[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct2[pi2[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, predicted class, target class)
            stats1.append((correct1.cpu(), pred[:, 4].cpu(), pred[:, 6].cpu(), tcls1))
            stats2.append((correct2.cpu(), pred[:, 5].cpu(), pred[:, 7].cpu(), tcls2))  # edit

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names1, names2), daemon=True).start()  # edit
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names1, names2), daemon=True).start()  # edit

    # Compute statistics
    stats1 = [np.concatenate(x, 0) for x in zip(*stats1)]  # to numpy
    stats2 = [np.concatenate(x, 0) for x in zip(*stats2)]  # to numpy  # edit

    # s1 = [np.count_nonzero(ba) for ba in stats1[0]]  # todo
    # s2 = [np.count_nonzero(ba) for ba in stats2[0]]  # todo
    # print("stats 1:", np.sum(s1), stats1[1:])  # todo
    # print("stats 2:", np.sum(s2), stats2[1:])  # todo

    if len(stats1) and stats1[0].any():
        p_1, r_1, ap_1, f1_1, ap_class_1 = ap_per_class(*stats1, plot=plots, save_dir=save_dir, names=names1, suffix='_1')
        ap50_1, ap_1 = ap_1[:, 0], ap_1.mean(1)  # AP@0.5, AP@0.5:0.95
        mp_1, mr_1, map50_1, map_1 = p_1.mean(), r_1.mean(), ap50_1.mean(), ap_1.mean()
        nt_1 = np.bincount(stats1[3].astype(np.int64), minlength=nc1)  # number of targets per class
    else:
        nt_1 = torch.zeros(1)
    if len(stats2) and stats2[0].any():
        p_2, r_2, ap_2, f1_2, ap_class_2 = ap_per_class(*stats2, plot=plots, save_dir=save_dir, names=names2, suffix='_2')  # edit
        ap50_2, ap_2 = ap_2[:, 0], ap_2.mean(1)  # AP@0.5, AP@0.5:0.95  # edit
        mp_2, mr_2, map50_2, map_2 = p_2.mean(), r_2.mean(), ap50_2.mean(), ap_2.mean()  # edit
        nt_2 = np.bincount(stats2[3].astype(np.int64), minlength=nc2)  # number of targets per class  # edit
    else:
        nt_2 = torch.zeros(1)  # edit

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all_1', seen, nt_1.sum(), mp_1, mr_1, map50_1, map_1))
    print(pf % ('all_2', seen, nt_2.sum(), mp_2, mr_2, map50_2, map_2))  # edit

    # Print results per class
    if (verbose or (nc1 < 50 and not training)) and nc1 > 1 and len(stats1):
        for i, c in enumerate(ap_class_1):
            print(pf % (names1[c], seen, nt_1[c], p_1[i], r_1[i], ap50_1[i], ap_1[i]))
    if (verbose or (nc2 < 50 and not training)) and nc2 > 1 and len(stats2):  # edit
        for i, c in enumerate(ap_class_2):
            print(pf % (names2[c], seen, nt_2[c], p_2[i], r_2[i], ap50_2[i], ap_2[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix1.plot(save_dir=save_dir, names=list(names1.values()), suffix='_1')
        confusion_matrix2.plot(save_dir=save_dir, names=list(names2.values()), suffix='_2')  # edit
        if wandb and wandb.run:
            val_batches = [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb.log({"Images": wandb_images, "Validation": val_batches}, commit=False)

    # Save JSON
    if save_json and len(jdict):  # todo ~
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps_1 = np.zeros(nc1) + map_1
    for i, c in enumerate(ap_class_1):
        maps_1[c] = ap_1[i]
    maps_2 = np.zeros(nc2) + map_2
    for i, c in enumerate(ap_class_2):
        maps_2[c] = ap_2[i]
    res_loss = (loss.cpu() / len(dataloader)).tolist()
    return maps_1, maps_2, t,\
        (mp_1, mr_1, map50_1, map_1, *res_loss),\
        (mp_2, mr_2, map50_2, map_2, *res_loss)


"""
Example run:
python test.py --data data/shape_ds.yaml --img 600 --conf 0.001 --iou 0.65 --weights weights/best.pt --batch 64 --task test
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot  # todo ~
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                _, _, t, r1, r2 = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r1 + r2 + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
