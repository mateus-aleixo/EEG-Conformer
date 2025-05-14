"""
EEG Conformer

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""

from classes.ExP import ExP
import datetime
import numpy as np
import os
import random
import time
import torch
from torch.backends import cudnn

gpus = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
cudnn.benchmark = False
cudnn.deterministic = True
# writer = SummaryWriter('./TensorBoardX/')


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()

        seed_n = np.random.randint(2021)
        print("seed is " + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print("Subject %d" % (i + 1))
        exp = ExP(i + 1, gpus)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print("THE BEST ACCURACY IS " + str(bestAcc))
        result_write.write(
            "Subject " + str(i + 1) + " : " + "Seed is: " + str(seed_n) + "\n"
        )
        result_write.write(
            "Subject "
            + str(i + 1)
            + " : "
            + "The best accuracy is: "
            + str(bestAcc)
            + "\n"
        )
        result_write.write(
            "Subject "
            + str(i + 1)
            + " : "
            + "The average accuracy is: "
            + str(averAcc)
            + "\n"
        )

        endtime = datetime.datetime.now()
        print("subject %d duration: " % (i + 1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

    best = best / 9
    aver = aver / 9

    result_write.write("**The average Best accuracy is: " + str(best) + "\n")
    result_write.write("The average Aver accuracy is: " + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
