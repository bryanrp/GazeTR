import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

def main(train, test, logfilename = 'test'):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.Model()

        statedict = torch.load(
                        os.path.join(modelpath, 
                            f"Iter_{saveiter}_{train.save.model_name}.pt"), 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )

        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset)

        # --- initialize accumulators ---
        errors = []
        errors_swapped = []
        count = 0

        logname = f"{saveiter}-{logfilename}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("filenames origins results gts cam_index frame_index\n")

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name' and key != 'filename' and key != 'cam_index' and key != 'frame_index':
                        data[key] = data[key].cuda()

                filenames = data["filename"]
                names = data["name"]
                gts = label.cuda()

                cam_indices = data["cam_index"]
                frame_indices = data["frame_index"]
           
                gazes = net(data)

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1                
                    err = gtools.angular(
                            gtools.gazeto3d(gaze),
                            gtools.gazeto3d(gt)
                        )
                    errors.append(err)

                    # Swap predicted gaze coordinates
                    gaze_swapped = np.array([gaze[1], gaze[0]])
                    err_sw = gtools.angular(
                        gtools.gazeto3d(gaze_swapped),
                        gtools.gazeto3d(gt)
                    )
                    errors_swapped.append(err_sw)
            
                    filename = [filenames[k]]
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    cam_index = cam_indices[k]
                    frame_index = frame_indices[k]
                    log = filename + name + [",".join(gaze)] + [",".join(gt)] + [cam_index] + [frame_index]
                    outfile.write(" ".join(log) + "\n")

            # after all samples, compute statistics
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            mean_err_sw = np.mean(errors_swapped)
            std_err_sw = np.std(errors_swapped)

            summary = (f"[{saveiter}] Total Num: {count}, "
                       f"mean: {mean_err:.4f}±{std_err:.4f}, "
                       f"mean_swapped: {mean_err_sw:.4f}±{std_err_sw:.4f}")
            outfile.write(summary)
            print(summary)
        outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')
    
    parser.add_argument('-l', '--logname', type=str,
                        help = 'log name', default = 'test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test, args.logname)
