import torch
import os
import sys
import argparse
import torch.nn.functional as F
from models import Resnet, R2Plus1D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from UCF101 import UCF101
from utils import args_print, printer, mean_confidence_interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, default="../UCF101FrameExtractor/UCF101_frames/")
    parser.add_argument("--labels-path", type=str, default="../UCF101FrameExtractor/UCF101_labels/")
    parser.add_argument("--tensorboard-path", type=str, default="./tensorboard/train1/")
    parser.add_argument("--list-number", type=int, default=1)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    # =============================================================
    # pad options
    parser.add_argument("--random-pad-sample", action="store_true")
    parser.add_argument("--pad-option", type=str, default="default")
    # frame options
    parser.add_argument("--uniform-frame-sample", action="store_true")
    parser.add_argument("--random-start-position", action="store_true")
    parser.add_argument("--max-interval", type=int, default=7)
    parser.add_argument("--random-interval", action="store_true")
    # =============================================================
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    args = parser.parse_args()

    # print args and save
    args_print(args)

    # write tensorboard
    writer = SummaryWriter(args.tensorboard_path)
    
    assert args.model in ["resnet", "r2plus1d"], "'{}' model is invalid !!".format(args.model)

    train_dataset = UCF101(
        model=args.model,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        list_number=args.list_number,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        train=True,
        # pad option
        random_pad_sample=args.random_pad_sample,
        pad_option=args.pad_option,
        # frame sampler option
        uniform_frame_sample=args.uniform_frame_sample,
        random_start_position=args.random_start_position,
        max_interval=args.max_interval,
        random_interval=args.random_interval,
    )

    test_dataset = UCF101(
        model=args.model,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        list_number=args.list_number,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        train=False,
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0 if os.name == 'nt' else 4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else 4)

    if args.model == "resnet":
        model = Resnet(
            num_classes=train_dataset.num_classes,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
        )
    if args.model == "r2plus1d":
        model = R2Plus1D(
            num_classes=train_dataset.num_classes,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    best = 0
    total_acc = 0
    total_loss = 0
    n_iter_train = 0
    n_iter_test = 0
    for e in range(1, args.num_epochs+1):
        train_acc = []
        train_loss = []
        model.train()
        for i, (datas, labels) in enumerate(train_loader):
            datas, labels = datas.to(device), labels.to(device)
            
            pred = model(datas)
            # calculate loss
            loss = F.cross_entropy(pred, labels)
            train_loss.append(loss.item())
            total_loss = sum(train_loss) / len(train_loss)

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            # print result
            printer("train", e, args.num_epochs, i+1, len(train_loader), loss.item(), total_loss, acc * 100, total_acc * 100)

            # tensorboard
            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1

        print("")
        test_acc = []
        test_loss = []
        model.eval()
        for i, (datas, labels) in enumerate(test_loader):
            datas, labels = datas.to(device), labels.to(device)
            
            pred = model(datas)

            # calculate loss
            loss = F.cross_entropy(pred, labels).item()
            test_loss.append(loss)
            total_loss = sum(test_loss) / len(test_loss)

            # calculate accuracy
            acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            test_acc.append(acc)
            total_acc = sum(test_acc) / len(test_acc)
            
            printer("test", e, args.num_epochs, i+1, len(test_loader), loss, total_loss, acc*100, total_acc*100)

            # tensorboard
            writer.add_scalar("Loss/test", loss, n_iter_test)
            writer.add_scalar("Accuracy/test", acc, n_iter_test)
            n_iter_test += 1
        
        # not saving
        if total_acc > best:
            best = total_acc
        m, h = mean_confidence_interval(test_acc)
        print("Best: {:.2f}% ({:.2f}+-{:.2f}) ".format(best*100, m*100, h))

        lr_scheduler.step()
